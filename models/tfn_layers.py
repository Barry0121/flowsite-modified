import time
from functools import partial

import torch_cluster
from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
import numpy as np
from e3nn.nn import BatchNorm
from models.pytorch_modules import LayerNorm, Linear, Encoder

ACTIVATIONS = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}
class ConvGraph():
    """
    Container class for graph convolution data.

    Stores edge indices, edge attributes, and spherical harmonic representations for use in
    tensor field network operations.

    Args:
        idx (torch.Tensor): Edge indices of shape [2, num_edges] containing source and destination node indices.
        attr (torch.Tensor): Edge attributes/features.
        sh (torch.Tensor): Spherical harmonic representations of edge vectors.
    """
    def __init__(self, idx, attr, sh):
        self.idx = idx
        self.attr = attr
        self.sh = sh

class RefinementTFNLayer(torch.nn.Module):
    """
    Refinement Tensor Field Network (TFN) Layer for molecular modeling.

    This layer implements a graph neural network with equivariant tensor field operations
    for refining the positions and features of ligand molecules in the context of a receptor.
    The layer handles ligand-ligand, ligand-receptor, and receptor-receptor interactions.

    Args:
        args: Configuration object containing the following parameters:
            ns (int): Number of scalar features.
            nv (int): Number of vector features.
            order (int): Maximum order of spherical harmonics for feature representations.
            sh_lmax (int): Maximum order of spherical harmonics for edge representations.
            lig_radius (float): Radius for ligand-ligand interactions.
            cross_radius (float): Radius for ligand-receptor interactions.
            radius_emb_dim (int): Dimension of radius embedding.
            fancy_init (bool): Whether to use special weight initialization.
            batch_norm (bool): Whether to use batch normalization.
            residual (bool): Whether to use residual connections.
            fc_dim (int): Hidden dimension of fully connected layers.
            faster (bool): Whether to use the faster tensor product implementation.
            no_tfn_rec2rec (bool): If True, disable receptor-receptor interactions.
            no_damping_factor (bool): If True, disable the damping factor for position updates.
            separate_update (bool): If True, use separate update for ligand nodes.
            fixed_lig_pos (bool): If True, fix ligand positions.
            update_last_when_fixed (bool): If True, update positions in the last layer even if fixed.
            feedforward (bool): Whether to use feedforward layers.
            pre_norm (bool): Whether to use layer normalization before feedforward.
            self_condition_bit (bool): Whether to use self-conditioning.
            post_norm (bool): Whether to use layer normalization after the update.
        last_layer (bool, optional): Whether this is the last layer in the network. Defaults to False.
    """
    def  __init__(self, args, last_layer=False):
        super().__init__()
        self.args = args
        self.last_layer = last_layer
        self.feature_irreps = o3.Irreps([(args.ns if l == 0 else args.nv, (l, 1)) for l in range(args.order + 1)])
        self.sh_irreps = o3.Irreps([(1, (l, 1)) for l in range(args.sh_lmax + 1)])

        self.lig_radius_embedding = nn.Sequential(
            GaussianSmearing(0.0, args.lig_radius, args.radius_emb_dim),
            Linear(
                args.radius_emb_dim,
                args.ns,
                init="relu" if args.fancy_init else "default",
            ),
            nn.ReLU(),
            Linear(args.ns, args.ns, init="final" if args.fancy_init else "default"),
        )

        self.cross_radius_embedding = nn.Sequential(
            GaussianSmearing(0.0, args.cross_radius, args.radius_emb_dim),
            Linear(
                args.radius_emb_dim,
                args.ns,
                init="relu" if args.fancy_init else "default",
            ),
            nn.ReLU(),
            Linear(args.ns, args.ns, init="final" if args.fancy_init else "default"),
        )

        parameters = dict(
            in_irreps=self.feature_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=self.feature_irreps,
            n_edge_features=3 * args.ns,
            batch_norm=args.batch_norm,
            residual=args.residual,
            hidden_features=args.fc_dim,
            faster=args.faster,
        )
        self.lig2lig = TensorProductConvLayer(**parameters)
        self.rec2lig = TensorProductConvLayer(**parameters)
        self.lig2rec = TensorProductConvLayer(**parameters)
        if not args.no_tfn_rec2rec:
            self.rec2rec = TensorProductConvLayer(**parameters)

        self.get_edge_attr = partial(get_edge_attr, ns=self.args.ns)
        if not args.no_damping_factor:
            self.damping_factor = nn.Parameter(torch.zeros(1))
        self.build_cg = partial(build_cg, sh_irreps=self.sh_irreps)

        if args.separate_update:
            self.linear_update = o3.Linear(self.feature_irreps, self.feature_irreps)

        if not args.fixed_lig_pos or last_layer and args.update_last_when_fixed:
            self.linear_out = o3.Linear(self.feature_irreps, "1x1e")

        if args.feedforward:
            self.lig_ff = nn.Sequential(
                LayerNorm(args.ns) if args.pre_norm else nn.Identity(),
                Linear(args.ns, 4 * args.ns, init="relu"),
                nn.ReLU(),
                Linear(4 * args.ns, args.ns, init="final"),
            )

            self.rec_ff = nn.Sequential(
                LayerNorm(args.ns) if args.pre_norm else nn.Identity(),
                Linear(args.ns, 4 * args.ns, init="relu"),
                nn.ReLU(),
                Linear(4 * args.ns, args.ns, init="final"),
            )
        if self.args.self_condition_bit:
            self.self_condition_bit_encoder = Encoder(emb_dim=args.ns, feature_dims=[2])

        if args.post_norm:
            self.lig_norm = LayerNorm(args.ns)
            self.rec_norm = LayerNorm(args.ns)

    def forward(self, data, rec_cg, lig_pos, lig_na, lig_ea, rec_na, temb=None, x_self=None, x_prior=None):
        start = time.time()
        if self.args.lig_coord_noise > 0: lig_pos = lig_pos + torch.randn_like(lig_pos) * self.args.lig_coord_noise
        lig_cg = self.build_cg(
            pos=lig_pos,
            edge_attr=lig_ea,
            edge_index=data["ligand", 'bond_edge', "ligand"].edge_index,
            radius_emb_func=self.lig_radius_embedding,
            batch=data["ligand"].batch,
            radius=self.args.lig_radius,
            pos_self=x_self,
            pos_prior=x_prior
        )

        cross_cg = self.build_cg(
            pos=(data["protein"].pos, lig_pos), # with this the ligand indices will be cross.cg[0]
            edge_attr=None,
            edge_index=None,
            radius_emb_func=self.cross_radius_embedding,
            batch=(data["protein"].batch, data["ligand"].batch),
            radius=self.args.cross_radius,
            time_embedding=None,
            pos_self=(data["protein"].pos, x_self),
            pos_prior=(data["protein"].pos, x_prior)
        )
        data.logs['radius_graph_time'] += time.time() - start
        lig_na_start, rec_na_start = lig_na, rec_na
        if self.args.time_condition_tfn and self.args.time_condition_repeat:
            lig_na[:,:self.args.ns] = lig_na[:,:self.args.ns] + temb[data['ligand'].batch.long()]
            rec_na[:,:self.args.ns] = rec_na[:,:self.args.ns] + temb[data['protein'].batch.long()]
            lig_cg.attr[:, :self.args.ns] = lig_cg.attr[:, :self.args.ns] + temb[data['ligand'].batch.long()[lig_cg.idx[0].long()]]
            rec_cg.attr[:, :self.args.ns] = rec_cg.attr[:, :self.args.ns] + temb[data['protein'].batch.long()[rec_cg.idx[0].long()]]
            cross_cg.attr[:, :self.args.ns] = cross_cg.attr[:, :self.args.ns] + temb[data['ligand'].batch.long()[cross_cg.idx[0].long()]]
        if self.args.self_condition_bit:
            lig_cg.attr[:, :self.args.ns] = lig_cg.attr[:, :self.args.ns] + self.self_condition_bit_encoder(data.self_condition_bit[data['ligand'].batch.long()[lig_cg.idx[0].long()]])
            cross_cg.attr[:, :self.args.ns] = cross_cg.attr[:, :self.args.ns] + self.self_condition_bit_encoder(data.self_condition_bit[data['ligand'].batch.long()[cross_cg.idx[0].long()]])

        if self.args.feedforward:
            lig_na = torch.cat([lig_na[:, : self.args.ns] + self.lig_ff(lig_na[:, : self.args.ns]),lig_na[:, self.args.ns:]],-1,)
            rec_na = torch.cat([rec_na[:, : self.args.ns] + self.rec_ff(rec_na[:, : self.args.ns]),rec_na[:, self.args.ns:]],-1,)
        if self.args.post_norm:
            lig_na = torch.cat([self.lig_norm(lig_na[:, : self.args.ns]), lig_na[:, self.args.ns:]],-1,)
            rec_na = torch.cat([self.rec_norm(rec_na[:, : self.args.ns]), rec_na[:, self.args.ns:]],-1,)

        edge_attr_ = self.get_edge_attr(cross_cg, lig_na, rec_na)
        lig2rec_na = self.lig2rec(lig_na, cross_cg.idx.flip(0), edge_attr_, cross_cg.sh, out_nodes=rec_na.shape[0], )
        if self.args.tfn_straight_combine: rec_na = rec_na + lig2rec_na

        if self.args.no_tfn_rec2rec:
            rec2rec_na = 0
        else:
            edge_attr_ = self.get_edge_attr(rec_cg, rec_na)
            rec2rec_na = self.rec2rec(rec_na, rec_cg.idx, edge_attr_, rec_cg.sh)
            if self.args.tfn_straight_combine: rec_na = rec_na + rec2rec_na

        edge_attr_ = self.get_edge_attr(cross_cg, lig_na, rec_na)
        rec2lig_na = self.rec2lig(rec_na, cross_cg.idx, edge_attr_, cross_cg.sh, out_nodes=lig_na.shape[0])
        if self.args.tfn_straight_combine: lig_na = lig_na + rec2lig_na

        edge_attr_ = self.get_edge_attr(lig_cg, lig_na)
        lig2lig_na = self.lig2lig(lig_na, lig_cg.idx, edge_attr_, lig_cg.sh)


        rec_na = rec_na_start + lig2rec_na + rec2rec_na
        if self.args.separate_update:
            lig_na = lig_na_start + self.linear_update(lig2lig_na + rec2lig_na)
        else:
            lig_na = lig_na_start + lig2lig_na + rec2lig_na

        if not self.args.fixed_lig_pos or self.last_layer and self.args.update_last_when_fixed:
            update = self.linear_out(lig_na)
            if not self.args.no_damping_factor:
                update = update * F.softplus(self.damping_factor)
            lig_pos = lig_pos + update

        return lig_pos, lig_na, rec_na

class TensorProductConvLayer(torch.nn.Module):
    """
    Tensor Product Convolution Layer for equivariant graph neural networks.

    Implements a graph convolution that preserves SO(3) equivariance using tensor products
    between node features and spherical harmonic edge features.

    Args:
        in_irreps (o3.Irreps): Input irreducible representations.
        sh_irreps (o3.Irreps): Spherical harmonic irreducible representations.
        out_irreps (o3.Irreps): Output irreducible representations.
        n_edge_features (int): Number of edge features.
        residual (bool, optional): Whether to use residual connections. Defaults to True.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        layer_norm (bool, optional): Whether to use layer normalization. Defaults to False.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        hidden_features (int, optional): Number of hidden features in the edge network.
                                         Defaults to n_edge_features if None.
        faster (bool, optional): Whether to use the faster tensor product implementation. Defaults to False.
    """
    def __init__(
        self,
        in_irreps,
        sh_irreps,
        out_irreps,
        n_edge_features,
        residual=True,
        batch_norm=False,
        layer_norm=False,
        dropout=0.0,
        hidden_features=None,
        faster=False
    ):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        if faster:
            print("Faster Tensor Product")
            self.tp = FasterTensorProduct(in_irreps, sh_irreps, out_irreps)
        else:
            self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.LayerNorm(n_edge_features) if layer_norm else nn.Identity(),
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, self.tp.weight_numel),
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(
        self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce="mean"
    ):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out

def get_edge_attr(conv_graph, node_attr1, node_attr2=None, ns=None):
    """
    Constructs edge attributes by concatenating edge features with node features.

    Args:
        conv_graph (ConvGraph): Graph container with edge indices and attributes.
        node_attr1 (torch.Tensor): Features of source nodes.
        node_attr2 (torch.Tensor, optional): Features of destination nodes.
                                             If None, use node_attr1. Defaults to None.
        ns (int, optional): Number of scalar features to use from node attributes. Defaults to None.

    Returns:
        torch.Tensor: Concatenated edge attributes.
    """
    if node_attr2 is None:
        node_attr2 = node_attr1
    src, dst = conv_graph.idx
    return torch.cat([conv_graph.attr, node_attr1[src, :ns], node_attr2[dst, :ns]], -1)

class GaussianSmearing(torch.nn.Module):
    """
    Gaussian smearing module for embedding distances.

    Transforms scalar distances into a vector of Gaussian basis functions.

    Args:
        start (float, optional): Minimum distance. Defaults to 0.0.
        stop (float, optional): Maximum distance. Defaults to 5.0.
        num_gaussians (int, optional): Number of Gaussian basis functions. Defaults to 50.
    """
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1) + 1E-6
        return torch.exp(self.coeff * torch.pow(dist, 2))

def build_cg(
    pos,
    edge_attr,
    edge_index,
    radius_emb_func,
    sh_irreps="1x0e+1x1e",
    batch=None,
    radius=None,
    time_embedding=None,
    pos_self=None,
    pos_prior=None
):
    """
    Builds a ConvGraph object for tensor field network operations.

    Creates edge attributes and spherical harmonic representations for edges in a graph.
    Can handle both single-graph and cross-graph (bipartite) scenarios.

    Args:
        pos (torch.Tensor or tuple): Node positions. If tuple, (pos1, pos2) for bipartite graphs.
        edge_attr (torch.Tensor): Edge attributes, can be None.
        edge_index (torch.Tensor): Edge indices, can be None.
        radius_emb_func (callable): Function to embed distances.
        sh_irreps (str, optional): Spherical harmonic irreps string. Defaults to "1x0e+1x1e".
        batch (torch.Tensor or tuple, optional): Batch indices for nodes. If tuple, (batch1, batch2). Defaults to None.
        radius (float or tuple, optional): Radius for neighborhood construction.
                                          If tuple, (radius1, radius2). Defaults to None.
        time_embedding (torch.Tensor, optional): Time embeddings to add to edge features. Defaults to None.
        pos_self (torch.Tensor or tuple, optional): Self positions for additional edge features. Defaults to None.
        pos_prior (torch.Tensor or tuple, optional): Prior positions for additional edge features. Defaults to None.

    Returns:
        ConvGraph: Graph container with edge indices, attributes, and spherical harmonics.
    """
    if isinstance(pos, tuple):
        pos1, pos2 = pos
        batch1, batch2 = batch or (None, None)
    else:
        pos1 = pos2 = pos
        batch1 = batch2 = batch

    if radius is not None:
        if isinstance(radius, tuple):
            radius1, radius2 = radius
        else:
            radius1 = radius2 = radius

        radius_edges = torch_cluster.radius(
            pos1 / radius1, pos2 / radius2, 1.0, batch1, batch2, max_num_neighbors=10000
        )

        if edge_index is not None:
            edge_index = torch.cat([edge_index, radius_edges], 1).long()
        else:
            edge_index = radius_edges

        if edge_attr is not None:
            edge_attr = F.pad(edge_attr, (0, 0, 0, radius_edges.shape[-1]))

    src, dst = edge_index
    edge_vec = pos1[dst.long()] - pos2[src.long()]

    if edge_attr is not None:
        edge_attr = edge_attr + radius_emb_func(edge_vec.norm(dim=-1))
    else:
        edge_attr = radius_emb_func(edge_vec.norm(dim=-1))


    if isinstance(pos_self, tuple):
        pos1_self, pos2_self = pos_self
    else:
        pos1_self = pos2_self = pos_self
    if pos1_self is not None and pos2_self is not None:
        edge_vec_self = pos1_self[dst.long()] - pos2_self[src.long()]
        edge_attr = edge_attr + radius_emb_func(edge_vec_self.norm(dim=-1))

    if isinstance(pos_prior, tuple):
        pos1_prior, pos2_prior = pos_prior
    else:
        pos1_prior = pos2_prior = pos_prior
    if pos1_prior is not None and pos2_prior is not None:
        edge_vec_prior = pos1_prior[dst.long()] - pos2_prior[src.long()]
        edge_attr = edge_attr + radius_emb_func(edge_vec_prior.norm(dim=-1))

    if time_embedding is not None:
        edge_attr = edge_attr + time_embedding[src.long()]

    edge_sh = o3.spherical_harmonics(
        sh_irreps, edge_vec, normalize=True, normalization="component"
    ).float()
    return ConvGraph(edge_index, edge_attr, edge_sh)


class FasterTensorProduct(torch.nn.Module):
    """
    Optimized tensor product implementation for SO(3)-equivariant neural networks.

    This is a faster implementation of the tensor product operation that works with
    spherical harmonic representations up to first order (L=1).

    Args:
        in_irreps (o3.Irreps): Input irreducible representations.
        sh_irreps (o3.Irreps): Spherical harmonic irreducible representations (must be 1x0e+1x1o).
        out_irreps (o3.Irreps): Output irreducible representations.
        **kwargs: Additional keyword arguments (unused).

    Note:
        This implementation only supports irreps with L=0 and L=1 for both input and output.
        The spherical harmonics must be first order (L=0,1).
    """
    def __init__(self, in_irreps, sh_irreps, out_irreps, **kwargs):
        super().__init__()
        #for ir in in_irreps:
        #    m, (l, p) = ir
        #    assert l in [0, 1], "Higher order in irreps are not supported"
        #for ir in out_irreps:
        #    m, (l, p) = ir
        #    assert l in [0, 1], "Higher order out irreps are not supported"
        assert o3.Irreps(sh_irreps) == o3.Irreps('1x0e+1x1o'), "sh_irreps don't look like 1st order spherical harmonics"
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)

        in_muls = {'0e': 0, '1o': 0, '1e': 0, '0o': 0}
        out_muls = {'0e': 0, '1o': 0, '1e': 0, '0o': 0}
        for (m, ir) in self.in_irreps: in_muls[str(ir)] = m
        for (m, ir) in self.out_irreps: out_muls[str(ir)] = m

        self.weight_shapes = {
            '0e': (in_muls['0e'] + in_muls['1o'], out_muls['0e']),
            '1o': (in_muls['0e'] + in_muls['1o'] + in_muls['1e'], out_muls['1o']),
            '1e': (in_muls['1o'] + in_muls['1e'] + in_muls['0o'], out_muls['1e']),
            '0o': (in_muls['1e'] + in_muls['0o'], out_muls['0o'])
        }
        self.weight_numel = sum(a * b for (a, b) in self.weight_shapes.values())

    def forward(self, in_, sh, weight):
        in_dict, out_dict = {}, {'0e': [], '1o': [], '1e': [], '0o': []}
        for (m, ir), sl in zip(self.in_irreps, self.in_irreps.slices()):
            in_dict[str(ir)] = in_[..., sl]
            if ir[0] == 1: in_dict[str(ir)] = in_dict[str(ir)].reshape(list(in_dict[str(ir)].shape)[:-1] + [-1, 3])
        sh_0e, sh_1o = sh[..., 0], sh[..., 1:]
        if '0e' in in_dict:
            out_dict['0e'].append(in_dict['0e'] * sh_0e.unsqueeze(-1))
            out_dict['1o'].append(in_dict['0e'].unsqueeze(-1) * sh_1o.unsqueeze(-2))
        if '1o' in in_dict:
            out_dict['0e'].append((in_dict['1o'] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3))
            out_dict['1o'].append(in_dict['1o'] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict['1e'].append(torch.linalg.cross(in_dict['1o'], sh_1o.unsqueeze(-2), dim=-1) / np.sqrt(2))
        if '1e' in in_dict:
            out_dict['1o'].append(torch.linalg.cross(in_dict['1e'], sh_1o.unsqueeze(-2), dim=-1) / np.sqrt(2))
            out_dict['1e'].append(in_dict['1e'] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict['0o'].append((in_dict['1e'] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3))
        if '0o' in in_dict:
            out_dict['1e'].append(in_dict['0o'].unsqueeze(-1) * sh_1o.unsqueeze(-2))
            out_dict['0o'].append(in_dict['0o'] * sh_0e.unsqueeze(-1))

        weight_dict = {}
        start = 0
        for key in self.weight_shapes:
            in_, out = self.weight_shapes[key]
            weight_dict[key] = weight[..., start:start + in_ * out].reshape(
                list(weight.shape)[:-1] + [in_, out]) / np.sqrt(in_)
            start += in_ * out

        if out_dict['0e']:
            out_dict['0e'] = torch.cat(out_dict['0e'], dim=-1)
            out_dict['0e'] = torch.matmul(out_dict['0e'].unsqueeze(-2), weight_dict['0e']).squeeze(-2)

        if out_dict['1o']:
            out_dict['1o'] = torch.cat(out_dict['1o'], dim=-2)
            out_dict['1o'] = (out_dict['1o'].unsqueeze(-2) * weight_dict['1o'].unsqueeze(-1)).sum(-3)
            out_dict['1o'] = out_dict['1o'].reshape(list(out_dict['1o'].shape)[:-2] + [-1])

        if out_dict['1e']:
            out_dict['1e'] = torch.cat(out_dict['1e'], dim=-2)
            out_dict['1e'] = (out_dict['1e'].unsqueeze(-2) * weight_dict['1e'].unsqueeze(-1)).sum(-3)
            out_dict['1e'] = out_dict['1e'].reshape(list(out_dict['1e'].shape)[:-2] + [-1])

        if out_dict['0o']:
            out_dict['0o'] = torch.cat(out_dict['0o'], dim=-1)
            # out_dict['0o'] = (out_dict['0o'].unsqueeze(-1) * weight_dict['0o']).sum(-2)
            out_dict['0o'] = torch.matmul(out_dict['0o'].unsqueeze(-2), weight_dict['0o']).squeeze(-2)

        out = []
        for _, ir in self.out_irreps:
            out.append(out_dict[str(ir)])
        return torch.cat(out, dim=-1)
