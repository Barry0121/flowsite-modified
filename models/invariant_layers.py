import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean
import numpy as np

class EnergyPredictor(nn.Module):
    """
    Energy prediction component that leverages invariant layer outputs.

    This module uses the rich structural and interaction representations
    already computed by the invariant layers to predict binding energy.

    Args:
        args (object): Configuration object with model parameters
        fold_dim (int): Dimension of the network embeddings
    """
    def __init__(self, args, fold_dim):
        super(EnergyPredictor, self).__init__()
        self.args = args

        # Global attention pooling for protein features
        self.protein_attention = nn.Sequential(
            nn.Linear(fold_dim, fold_dim // 2),
            nn.ReLU(),
            nn.Linear(fold_dim // 2, 1)
        )

        # Global attention pooling for ligand features
        self.ligand_attention = nn.Sequential(
            nn.Linear(fold_dim, fold_dim // 2),
            nn.ReLU(),
            nn.Linear(fold_dim // 2, 1)
        )

        # Optional: Cross-attention for interaction features
        self.use_cross_features = args.use_cross_features if hasattr(args, 'use_cross_features') else True
        if self.use_cross_features:
            self.cross_attention = nn.Sequential(
                nn.Linear(fold_dim, fold_dim // 2),
                nn.ReLU(),
                nn.Linear(fold_dim // 2, 1)
            )

        # Feature combination and final prediction
        input_dim = fold_dim * 2
        if self.use_cross_features:
            input_dim += fold_dim

        self.confidence_branch = args.confidence_branch if hasattr(args, 'confidence_branch') else False

        # Main energy prediction MLP
        self.energy_mlp = nn.Sequential(
            nn.Linear(input_dim, fold_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fold_dim, fold_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Final output layers
        self.energy_output = nn.Linear(fold_dim // 2, 1)

        # Optional confidence score branch
        if self.confidence_branch:
            self.confidence_output = nn.Sequential(
                nn.Linear(fold_dim // 2, fold_dim // 4),
                nn.ReLU(),
                nn.Linear(fold_dim // 4, 1),
                nn.Sigmoid()  # Confidence score between 0 and 1
            )

    def attention_pooling(self, node_features, attention_network, batch_indices, num_graphs):
        """
        Apply attention-based pooling to node features.

        Args:
            node_features (torch.Tensor): Features per node
            attention_network (nn.Module): Network to compute attention weights
            batch_indices (torch.Tensor): Batch assignment for each node
            num_graphs (int): Number of graphs in the batch

        Returns:
            torch.Tensor: Pooled features
        """
        # Calculate attention scores
        attention_scores = attention_network(node_features)
        # Apply softmax per graph
        attention_weights = scatter_softmax(attention_scores, batch_indices, dim=0)
        # Weighted average
        weighted_features = node_features * attention_weights
        pooled_features = scatter_sum(weighted_features, batch_indices, dim=0, dim_size=num_graphs)
        return pooled_features

    def forward(self, rec_na, lig_na, cross_ea, cross_idx, protein_batch, ligand_batch):
        """
        Forward pass of the EnergyPredictor.

        Args:
            rec_na (torch.Tensor): Protein node features from invariant layers
            lig_na (torch.Tensor): Ligand node features from invariant layers
            cross_ea (torch.Tensor): Cross-edge features from invariant layers
            cross_idx (torch.Tensor): Cross-edge indices
            protein_batch (torch.Tensor): Batch indices for protein nodes
            ligand_batch (torch.Tensor): Batch indices for ligand nodes

        Returns:
            tuple: (energy_prediction, confidence_score) if confidence_branch=True
                  energy_prediction otherwise
        """
        # Get the number of graphs in the batch
        num_graphs = ligand_batch.max().item() + 1

        # Apply attention pooling to protein and ligand features
        protein_pooled = self.attention_pooling(
            rec_na,
            self.protein_attention,
            protein_batch,
            num_graphs
        )

        ligand_pooled = self.attention_pooling(
            lig_na,
            self.ligand_attention,
            ligand_batch,
            num_graphs
        )

        # Combine features
        if self.use_cross_features and cross_ea is not None:
            # Pool cross-edge features
            complex_ids = ligand_batch[cross_idx[0]]
            cross_pooled = self.attention_pooling(
                cross_ea,
                self.cross_attention,
                complex_ids,
                num_graphs
            )
            # Concatenate all features
            combined_features = torch.cat([protein_pooled, ligand_pooled, cross_pooled], dim=1)
        else:
            # Just use protein and ligand features
            combined_features = torch.cat([protein_pooled, ligand_pooled], dim=1)

        # Process through MLP
        features = self.energy_mlp(combined_features)
        energy = self.energy_output(features)

        if self.confidence_branch:
            confidence = self.confidence_output(features)
            return energy, confidence
        else:
            return energy
        
class InvariantLayer(nn.Module):
    def __init__(self, args, device, update_edges=True):
        super(InvariantLayer, self).__init__()
        self.args = args
        self.device = device
        fold_dim = args.fold_dim
        self.rec2rec = PiFoldConv(args, fold_dim, fold_dim * 2, dropout=args.inv_dropout, update_edges=update_edges)
        if not self.args.ignore_lig:
            self.lig2rec = PiFoldConv(args, fold_dim, fold_dim * 2, dropout=args.inv_dropout, update_edges=update_edges)
            self.rec2lig = PiFoldConv(args, fold_dim, fold_dim * 2, dropout=args.inv_dropout, update_edges=update_edges)
            self.lig2lig = PiFoldConv(args, fold_dim, fold_dim * 2, dropout=args.inv_dropout, update_edges=update_edges)

    def forward(self, data, rec_idx, rec_na, rec_ea, lig_idx, lig_na, lig_ea, cross_idx, cross_ea, temb):
        if self.args.time_condition_inv and self.args.time_condition_repeat:
            rec_nap = rec_na + temb[data['protein'].batch.long()]
            lig_nap = lig_na + temb[data['ligand'].batch.long()]
            lig_eap = lig_ea + temb[data['ligand'].batch.long()[lig_idx[0].long()]]
            rec_eap = rec_ea + temb[data['protein'].batch.long()[rec_idx[0].long()]]
            cross_eap = cross_ea + temb[data['ligand'].batch.long()[cross_idx[0].long()]]
        else:
            rec_nap, lig_nap, lig_eap, rec_eap, cross_eap = rec_na, lig_na, lig_ea, rec_ea, cross_ea

        if not self.args.ignore_lig:
            rec2lig_na, rec2lig_ea = self.rec2lig(rec_nap, lig_nap, cross_eap, cross_idx, data['ligand'].batch)
            if self.args.inv_straight_combine: lig_nap = lig_nap + rec2lig_na
            lig2lig_na, lig2lig_ea = self.lig2lig(lig_nap, lig_nap, lig_eap, lig_idx, data['ligand'].batch)
            if self.args.inv_straight_combine: lig_nap = lig_nap + lig2lig_na
            lig2rec_na, lig2rec_ea = self.lig2rec(lig_nap, rec_nap, cross_eap, cross_idx.flip(0), data['protein'].batch)
            if self.args.inv_straight_combine: rec_nap = rec_nap + lig2rec_na
        rec2rec_na, rec2rec_ea = self.rec2rec(rec_nap, rec_nap, rec_eap, rec_idx, data['protein'].batch)

        rec_na = rec2rec_na
        rec_ea = rec2rec_ea
        if not self.args.ignore_lig:
            rec_na = rec_na + lig2rec_na
            lig_na = lig2lig_na + rec2lig_na
            lig_ea = lig2lig_ea
            cross_ea = lig2rec_ea + rec2lig_ea
        return rec_na, rec_ea, lig_na, lig_ea, cross_ea

# The following classes were adapted from https://github.com/A4Bio/PiFold/blob/main/methods/prodesign_module.py

class PiFoldConv(nn.Module):
    def __init__(self, args, num_hidden, num_in, dropout=0.1, scale=30, update_edges=True):
        super(PiFoldConv, self).__init__()
        self.args = args
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads=4)
        self.update_edegs = update_edges

        if update_edges:
            self.edge_net = EdgeMLP(num_hidden, num_in)

        if args.node_context or args.edge_context:
            self.context = Context(num_hidden, num_in, node_context=args.node_context, edge_context=args.edge_context)

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden)
        )

    def forward(self, src_na, dst_na, ea, edge_idx, batch_id):
        dh = self.attention(src_na, dst_na, ea, edge_idx)

        dst_na_update = self.norm[0](dst_na + self.dropout(dh))
        dh = self.dense(dst_na_update)
        dst_na_update = self.norm[1](dst_na_update + self.dropout(dh))

        if self.update_edegs:
            ea = self.edge_net(src_na, dst_na_update, ea, edge_idx)

        if self.args.node_context or self.args.edge_context:
            dst_na_update, ea = self.context(dst_na_update, ea, edge_idx, batch_id)
        return dst_na_update, ea






#################################### node modules ###############################
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=True):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp

        self.W_V = nn.Sequential(nn.Linear(num_in, num_hidden),
                                 nn.GELU(),
                                 nn.Linear(num_hidden, num_hidden),
                                 nn.GELU(),
                                 nn.Linear(num_hidden, num_hidden)
                                 )
        self.Bias = nn.Sequential(
            nn.Linear(num_hidden * 3, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_heads)
        )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, src_na, dst_na, ea, edge_idx):
        dst_idx = edge_idx[0] # torch_cluster.radius puts src and dst this way around. Double checked that this is right
        src_idx = edge_idx[1]

        d = int(self.num_hidden / self.num_heads)

        w = self.Bias(torch.cat([src_na[src_idx], ea, dst_na[dst_idx]], dim=-1)).view(ea.shape[0], self.num_heads, 1)
        attend_logits = w / np.sqrt(d)

        V = self.W_V(torch.cat([src_na[src_idx], ea], dim=-1)).view(-1, self.num_heads, d)
        attend = scatter_softmax(attend_logits, index=dst_idx, dim=0)
        dst_na_update = scatter_sum(attend * V, dst_idx, dim=0, dim_size=len(dst_na)).view([-1, self.num_hidden])

        if self.output_mlp:
            dst_na_update = self.W_O(dst_na_update)
        else:
            dst_na_update = dst_na_update
        return dst_na_update


#################################### edge modules ###############################
class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(EdgeMLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, src_na, dst_na, ea, edge_idx):
        dst_idx = edge_idx[0] # torch_cluster.radius puts src and dst this way around
        src_idx = edge_idx[1]

        h_EV = torch.cat([src_na[src_idx], ea, dst_na[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        ea_update = self.norm(ea + self.dropout(h_message))
        return ea_update


#################################### context modules ###############################
class Context(nn.Module):
    def __init__(self, num_hidden, num_in, scale=30, node_context=False,
                 edge_context=False):
        super(Context, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context
        if self.node_context:
            self.V_MLP_g = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.Sigmoid()
            )

        if self.edge_context:
            self.E_MLP = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden)
            )

            self.E_MLP_g = nn.Sequential(
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.Sigmoid()
            )

    def forward(self, dst_na, ea, edge_idx, batch_id):
        if self.node_context:
            c_V = scatter_mean(dst_na, batch_id, dim=0)
            dst_na = dst_na * self.V_MLP_g(c_V[batch_id])

        if self.edge_context:
            c_V = scatter_mean(dst_na, batch_id, dim=0)
            ea = ea * self.E_MLP_g(c_V[batch_id[edge_idx[0]]])

        return dst_na, ea