import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from models.tfn_layers import GaussianSmearing


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embeddings for time steps in diffusion models.

    This module creates positional embeddings using sine and cosine functions
    of different frequencies, similar to those in the Transformer architecture.

    Attributes:
        embedding_dim (int): Dimension of the embedding vector.
        max_positions (int): Maximum number of positions to be embedded.
        embedding_scale (float): Scaling factor for the time steps.

    Reference:
        https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    """
    def __init__(self, embedding_dim, embedding_scale, max_positions=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.embedding_scale = embedding_scale

    def forward(self, timesteps):
        """
        Compute sinusoidal embeddings for input timesteps.

        Args:
            timesteps (torch.Tensor): Tensor of shape [batch_size] containing time steps.

        Returns:
            torch.Tensor: Tensor of shape [batch_size, embedding_dim] containing
                the embedded time steps.

        Raises:
            AssertionError: If timesteps is not a 1D tensor.
        """
        assert len(timesteps.shape) == 1
        timesteps = timesteps * self.embedding_scale
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], self.embedding_dim)
        return emb


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.

    Projects input values into a higher dimensional space using random
    Fourier features with Gaussian weights.

    Attributes:
        W (nn.Parameter): Fourier basis frequencies sampled from a Gaussian.

    Reference:
        https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        """
        Project input values to Fourier space.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1].

        Returns:
            torch.Tensor: Projected tensor of shape [batch_size, embedding_size].
        """
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb

def get_time_mapping(embedding_type, embedding_dim, embedding_scale=10000):
    """
    Create a time embedding function based on the specified type.

    Args:
        embedding_type (str): The type of embedding to use. Options are:
            - 'sinusoidal': Uses position encoding style embeddings
            - 'fourier': Uses random Fourier features
            - 'gaussian': Uses Gaussian RBF embeddings
        embedding_dim (int): The dimension of the embedding
        embedding_scale (float, optional): Scale factor for embeddings. Defaults to 10000.

    Returns:
        nn.Module: The embedding function that takes time values and outputs embeddings

    Raises:
        NotImplementedError: If embedding_type is not one of the supported types
    """
    if embedding_type == 'sinusoidal':
        emb_func = SinusoidalEmbedding(embedding_dim=embedding_dim, embedding_scale=embedding_scale)
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    elif embedding_type == 'gaussian':
        emb_func = GaussianSmearing(0.0, 1, embedding_dim)
    else:
        raise NotImplemented
    return emb_func

def pyg_add_vp_noise(args, batch):
    """
    Add noise to ligand positions using a variance-preserving diffusion process.

    This function adds noise to ligand positions according to a diffusion process
    with variance preservation (VP), storing the original positions for later evaluation.

    Diffusion Equation: x_t = sqrt(α_t) · x_0 + sqrt(1 - α_t) · ε

    Args:
        args: Configuration arguments.
        batch: PyTorch Geometric batch object containing ligand data.

    Effects:
        - Stores original positions in batch["ligand"].shadow_pos
        - Updates batch["ligand"].pos with noised positions
        - Adds normalized_t, t, and std to batch for tracking diffusion parameters
    """
    # Store original positions for loss calculation during training
    batch["ligand"].shadow_pos = batch["ligand"].pos.clone()

    # Create SDE object using protein-specific sigma values that control noise levels
    sde = DiffusionSDE(batch.protein_sigma)

    # Sample random time points with quadratic schedule (biases toward earlier times)
    # This transforms uniform random values in [0,1] to t² for better training dynamics
    batch.normalized_t = torch.square(torch.rand_like(sde.lamb))

    # Scale normalized time to actual diffusion time range and add offset
    batch.t = sde.max_t() * batch.normalized_t + 1  # temporary solution

    # Apply mean drift component: scale positions by time-dependent factor
    # For VP diffusion, this factor decreases exponentially with time: e^(-λt/2)
    # The indexing ensures each atom gets the right factor for its molecule
    batch["ligand"].pos *= sde.mu_factor(batch.t)[batch["ligand"].batch, None]

    # Generate standard Gaussian noise with same shape as ligand positions
    noise = torch.randn_like(batch["ligand"].pos)

    # Calculate the standard deviation for noise scaling at current time points
    # In VP diffusion, this increases to compensate for the shrinking mean
    # Mathematical form: sqrt((1-e^(-λt))/λ)
    batch.std = torch.sqrt(sde.var(batch.t))

    # Add scaled noise to complete the forward diffusion:
    # x_t = sqrt(α_t)·x_0 + sqrt(1-α_t)·ε
    # where sqrt(α_t) is mu_factor and sqrt(1-α_t) is std
    batch["ligand"].pos += noise * batch.std[batch["ligand"].batch, None]

def pyg_add_harmonic_noise(args, batch):
    """Add noise to ligand positions using a harmonic diffusion process.

    This function:
    1. Computes the eigendecomposition of the graph Laplacian from bond edges
    2. Transforms ligand positions to the eigenbasis
    3. Adds noise to the transformed positions
    4. Transforms back to the original space

    Args:
        args: Configuration arguments, including:
            - prior_scale: Scaling factor for the noise
            - highest_noise_only: If True, uses maximum noise level
        batch: PyTorch Geometric batch object containing ligand data and bonds

    Effects:
        - Adds D, P matrices to batch for the eigendecomposition
        - Updates batch["ligand"].pos with noised positions
        - Adds t01, normalized_t, t, and std to batch for tracking diffusion parameters
    """
    # Get batch indices for each atom to keep track of which molecule it belongs to
    bid = batch['ligand'].batch

    # Create diffusion SDE with protein sigma scaled by prior_scale parameter
    # This controls the overall magnitude of the noise
    sde = DiffusionSDE(batch.protein_sigma * args.prior_scale)

    # Either use maximum noise level (t=1) or sample random time points
    if args.highest_noise_only:
        batch.t01 = torch.ones_like(sde.lamb)
    else:
        # Sample random time values between 0 and 1 for each molecule
        batch.t01 = torch.rand_like(sde.lamb)

    # Apply quadratic schedule to bias sampling toward earlier times
    batch.normalized_t = torch.square(batch.t01)

    # Scale normalized time to actual diffusion time range and add offset
    batch.t = sde.max_t() * batch.normalized_t + 1  # temporary solution

    # Extract bond edges from the molecular graph
    edges = batch['ligand', 'bond_edge', 'ligand'].edge_index

    # Deduplicate edges to get each bond only once (i→j but not j→i)
    edges = edges[:, edges[0] < edges[1]]

    # Calculate eigendecomposition of the graph Laplacian
    # D: eigenvalues representing vibrational modes of the molecule
    # P: eigenvectors forming an orthogonal basis for molecular movement
    D, P = HarmonicSDE.diagonalize(batch['ligand'].num_nodes, edges=edges.T, lamb=sde.lamb[bid], ptr=batch['ligand'].ptr)

    # Store decomposition in batch for later use (e.g., in reverse process)
    batch.D = D
    batch.P = P

    # Update lambda values with the eigenvalues
    sde.lamb = D

    # Transform atom positions to the eigenbasis using P^T
    # This decouples the correlated motions into independent components
    pos = P.T @ batch['ligand'].pos

    # Apply mean drift to each component in the eigenbasis
    # Similar to VP diffusion, but each mode decays at its own rate
    pos = pos * sde.mu_factor(batch.t[bid])[:, None]

    # Generate standard Gaussian noise with same shape as ligand positions
    noise = torch.randn_like(batch["ligand"].pos)

    # Calculate noise scaling based on the variance at current time points
    std = torch.sqrt(sde.var(batch.t[bid]))

    # Add scaled noise to the transformed positions
    # This adds noise proportionally to each vibrational mode
    pos += noise * std[:, None]

    # Transform back to Cartesian coordinates using P
    # This ensures the noise respects the molecular connectivity
    batch['ligand'].pos = P @ pos

    # Calculate average standard deviation across molecules for monitoring
    batch.std = scatter_mean(std ** 2, bid) ** 0.5

def sample_prior(batch, sigma, harmonic=True):
    """
    Sample ligand positions from prior distribution.

    Args:
        batch: PyTorch Geometric batch object containing ligand data and bonds
        sigma (float): Scale of the noise/prior
        harmonic (bool, optional): Whether to use harmonic prior (True) or Gaussian prior (False).
            Defaults to True.

    Returns:
        torch.Tensor: Sampled ligand positions tensor of shape [num_atoms, 3]

    Raises:
        Exception: If there's an error in building the graph Laplacian eigensystem
    """
    if harmonic:
        bid = batch['ligand'].batch
        sde = DiffusionSDE(batch.protein_sigma * sigma)

        edges = batch['ligand', 'bond_edge', 'ligand'].edge_index
        edges = edges[:, edges[0] < edges[1]]  # de-duplicate
        try:
            D, P = HarmonicSDE.diagonalize(batch['ligand'].num_nodes, edges=edges.T, lamb=sde.lamb[bid], ptr=batch['ligand'].ptr)
        except Exception as e:
            print('batch["ligand"].num_nodes', batch['ligand'].num_nodes)
            print("batch['ligand'].size", batch['ligand'].size)
            print("batch['protein'].size", batch['protein'].batch.bincount())
            print(batch.pdb_id)
            raise e
        noise = torch.randn_like(batch["ligand"].pos)
        prior = P @ (noise / torch.sqrt(D)[:, None])
        return prior
    else:
        prior = torch.randn_like(batch["ligand"].pos)
        return prior * sigma

class DiffusionSDE:
    """
    Stochastic Differential Equation for diffusion processes.

    This class implements an Ornstein-Uhlenbeck SDE for diffusion processes
    with customizable parameters for variance and time scaling.

    Attributes:
        lamb (torch.Tensor): Lambda parameter controlling noise level (1/σ²)
        tau_factor (float): Factor controlling the maximum time
    """
    def __init__(self, sigma: torch.Tensor, tau_factor=5.0):
        self.lamb = 1 / sigma**2
        self.tau_factor = tau_factor

    def var(self, t):
        """
        Calculate variance of the diffusion process at time t.

        Args:
            t (torch.Tensor): Time values

        Returns:
            torch.Tensor: Variance at time t
        """
        return (1 - torch.exp(-self.lamb * t)) / self.lamb

    def max_t(self):
        """
        Calculate maximum time for the diffusion process.

        Returns:
            torch.Tensor: Maximum time values
        """
        return self.tau_factor / self.lamb

    def mu_factor(self, t):
        """
        Calculate mean scaling factor at time t.

        Args:
            t (torch.Tensor): Time values

        Returns:
            torch.Tensor: Mean scaling factor at time t
        """
        return torch.exp(-self.lamb * t / 2)

class HarmonicSDE:
    """
    Harmonic Stochastic Differential Equation for structural diffusion.

    This class implements a structure-aware diffusion process that uses
    the graph Laplacian to define a multivariate Gaussian prior over molecular geometries,
    maintaining correlations between connected atoms during generation.

    Attributes:
        use_cuda (bool): Whether to use CUDA for computations
        l (int): Index offset for the first non-zero mode
        D (array-like): Eigenvalues of the graph Laplacian
        P (array-like): Eigenvectors of the graph Laplacian
        N (int): Number of nodes in the graph
    """
    def __init__(self, N=None, edges=[], antiedges=[], a=0.5, b=0.3,
                 J=None, diagonalize=True):
        """
        Initialize Harmonic SDE.

        Args:
            N (int, optional): Number of nodes. Defaults to None.
            edges (list, optional): List of edge tuples (i,j). Defaults to [].
            antiedges (list, optional): List of negative edge tuples. Defaults to [].
            a (float, optional): Edge weight parameter. Defaults to 0.5.
            b (float, optional): Anti-edge weight parameter. Defaults to 0.3.
            J (array-like, optional): Precomputed Laplacian matrix. Defaults to None.
            diagonalize (bool, optional): Whether to diagonalize immediately. Defaults to True.
        """
        self.use_cuda = False
        self.l = 1
        if not diagonalize: return
        if J is not None:
            J = J
            self.D, P = np.linalg.eigh(J)
            self.P = P
            self.N = self.D.size
            return


    @staticmethod
    def diagonalize(N, edges=[], antiedges=[], a=1, b=0.3, lamb=0., ptr=None):
        """
        Diagonalize the graph Laplacian matrix.

        Args:
            N (int): Number of nodes
            edges (list, optional): List of edge tuples (i,j). Defaults to [].
            antiedges (list, optional): List of negative edge tuples. Defaults to [].
            a (float, optional): Edge weight parameter. Defaults to 1.
            b (float, optional): Anti-edge weight parameter. Defaults to 0.3.
            lamb (float or torch.Tensor, optional): Additional diagonal term. Defaults to 0.
            ptr (torch.Tensor, optional): Pointers to subgraphs for batch processing. Defaults to None.

        Returns:
            tuple: (D, P) where D are eigenvalues and P are eigenvectors of the Laplacian
        """
        J = torch.zeros((N, N), device=edges.device)  # temporary fix
        for i, j in edges:
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a
        for i, j in antiedges:
            J[i, i] -= b
            J[j, j] -= b
            J[i, j] = J[j, i] = b
        J += torch.diag(lamb)
        if ptr is None:
            return torch.linalg.eigh(J)

        Ds, Ps = [], []
        for start, end in zip(ptr[:-1], ptr[1:]):
            D, P = torch.linalg.eigh(J[start:end, start:end])
            Ds.append(D)
            Ps.append(P)
        return torch.cat(Ds), torch.block_diag(*Ps)

    def eigens(self, t):
        """
        Calculate eigenvalues of the covariance matrix at time t.

        Args:
            t (float or torch.Tensor): Time value

        Returns:
            array-like: Eigenvalues of covariance matrix
        """
        np_ = torch if self.use_cuda else np
        D = 1 / self.D * (1 - np_.exp(-t * self.D))
        t = torch.tensor(t, device='cuda').float() if self.use_cuda else t
        return np_.where(D != 0, D, t)

    def conditional(self, mask, x2):
        """
        Sample from conditional distribution given partial coordinates.

        Args:
            mask (array-like): Boolean mask of known positions
            x2 (array-like): Known coordinates

        Returns:
            array-like: Sampled coordinates for unknown positions
        """
        J_11 = self.J[~mask][:, ~mask]
        J_12 = self.J[~mask][:, mask]
        h = -J_12 @ x2
        mu = np.linalg.inv(J_11) @ h
        D, P = np.linalg.eigh(J_11)
        z = np.random.randn(*mu.shape)
        return (P / D ** 0.5) @ z + mu

    def A(self, t, invT=False):
        """
        Calculate transformation matrix A at time t.

        Args:
            t (float or torch.Tensor): Time value
            invT (bool, optional): Whether to also return inverse transpose. Defaults to False.

        Returns:
            array-like or tuple: A matrix, or (A, A_inv_transpose) if invT=True
        """
        D = self.eigens(t)
        A = self.P * (D ** 0.5)
        if not invT: return A
        AinvT = self.P / (D ** 0.5)
        return A, AinvT

    def Sigma_inv(self, t):
        """
        Calculate inverse covariance matrix at time t.

        Args:
            t (float or torch.Tensor): Time value

        Returns:
            array-like: Inverse covariance matrix
        """
        D = 1 / self.eigens(t)
        return (self.P * D) @ self.P.T

    def Sigma(self, t):
        """
        Calculate covariance matrix at time t.

        Args:
            t (float or torch.Tensor): Time value

        Returns:
            array-like: Covariance matrix
        """
        D = self.eigens(t)
        return (self.P * D) @ self.P.T

    @property
    def J(self):
        """
        Calculate precision matrix (graph Laplacian).

        Returns:
            array-like: Precision matrix
        """
        return (self.P * self.D) @ self.P.T

    def rmsd(self, t):
        """
        Calculate expected RMSD at time t.

        Args:
            t (float): Time value

        Returns:
            float: Expected RMSD
        """
        l = self.l
        D = 1 / self.D * (1 - np.exp(-t * self.D))
        return np.sqrt(3 * D[l:].mean())

    def sample(self, t, x=None, score=False, k=None, center=True, adj=False):
        """
        Sample from the SDE at time t.

        Args:
            t (float): Time value
            x (array-like, optional): Initial position. Defaults to None.
            score (bool, optional): Whether to also return score. Defaults to False.
            k (int, optional): Number of modes to use. Defaults to None.
            center (bool, optional): Whether to center the sample. Defaults to True.
            adj (bool, optional): Whether to adjust the score. Defaults to False.

        Returns:
            array-like or tuple: Sampled positions, or (positions, score) if score=True
        """
        l = self.l
        np_ = torch if self.use_cuda else np
        if x is None:
            if self.use_cuda:
                x = torch.zeros((self.N, 3), device='cuda').float()
            else:
                x = np.zeros((self.N, 3))
        if t == 0: return x
        z = np.random.randn(self.N, 3) if not self.use_cuda else torch.randn(self.N, 3, device='cuda').float()
        D = self.eigens(t)
        xx = self.P.T @ x
        if center: z[0] = 0; xx[0] = 0
        if k: z[k + l:] = 0; xx[k + l:] = 0

        out = np_.exp(-t * self.D / 2)[:, None] * xx + np_.sqrt(D)[:, None] * z

        if score:
            score = -(1 / np_.sqrt(D))[:, None] * z
            if adj: score = score + self.D[:, None] * out
            return self.P @ out, self.P @ score
        return self.P @ out

    def score_norm(self, t, k=None, adj=False):
        """
        Calculate score norm at time t.

        Args:
            t (float): Time value
            k (int, optional): Number of modes to use. Defaults to None.
            adj (bool, optional): Whether to adjust the score. Defaults to False.

        Returns:
            float: Score norm
        """
        if k == 0: return 0
        l = self.l
        np_ = torch if self.use_cuda else np
        k = k or self.N - 1
        D = 1 / self.eigens(t)
        if adj: D = D * np_.exp(-self.D * t)
        return (D[l:k + l].sum() / self.N) ** 0.5

    def inject(self, t, modes):
        """
        Inject noise along specific modes.

        Args:
            t (float): Time value
            modes (array-like): Boolean mask of modes to inject noise into

        Returns:
            array-like: Noise along specified modes
        """
        # Returns noise along the given modes
        z = np.random.randn(self.N, 3) if not self.use_cuda else torch.randn(self.N, 3, device='cuda').float()
        z[~modes] = 0
        A = self.A(t, invT=False)
        return A @ z

    def score(self, x0, xt, t):
        """
        Calculate score (gradient of log-likelihood) at time t.

        Args:
            x0 (array-like): Initial position
            xt (array-like): Position at time t
            t (float): Time value

        Returns:
            array-like: Score at time t
        """
        # Score of the diffusion kernel
        Sigma_inv = self.Sigma_inv(t)
        mu_t = (self.P * np.exp(-t * self.D / 2)) @ (self.P.T @ x0)
        return Sigma_inv @ (mu_t - xt)

    def project(self, X, k, center=False):
        """
        Project coordinates onto the first k modes.

        Args:
            X (array-like): Coordinates to project
            k (int): Number of modes to project onto
            center (bool, optional): Whether to center the projection. Defaults to False.

        Returns:
            array-like: Projected coordinates
        """
        l = self.l
        # Projects onto the first k nonzero modes (and optionally centers)
        D = self.P.T @ X
        D[k + l:] = 0
        if center: D[0] = 0
        return self.P @ D

    def unproject(self, X, mask, k, return_Pinv=False):
        """
        Find coordinates along first k modes that best match a subset of positions.

        Args:
            X (array-like): Target coordinates (for masked positions)
            mask (array-like): Boolean mask of positions to match
            k (int): Number of modes to use
            return_Pinv (bool, optional): Whether to also return pseudoinverse. Defaults to False.

        Returns:
            array-like or tuple: Unprojected coordinates, or (coordinates, Pinv) if return_Pinv=True
        """
        # Finds the vector along the first k nonzero modes whose mask is closest to X
        l = self.l
        PP = self.P[mask, :k + l]
        Pinv = np.linalg.pinv(PP)
        out = self.P[:, :k + l] @ Pinv @ X
        if return_Pinv: return out, Pinv
        return out

    def energy(self, X):
        """
        Calculate energy of a configuration.

        Args:
            X (array-like): Coordinates

        Returns:
            array-like: Energy of configuration
        """
        l = self.l
        return (self.D[:, None] * (self.P.T @ X) ** 2).sum(-1)[l:] / 2

    @property
    def free_energy(self):
        """Calculate free energy of the system.

        Returns:
            float: Free energy
        """
        l = self.l
        return 3 * np.log(self.D[l:]).sum() / 2

    def KL_H(self, t):
        """Calculate KL divergence between equilibrium and time t distributions.

        Args:
            t (float): Time value

        Returns:
            float: KL divergence
        """
        l = self.l
        D = self.D[l:]
        return -3 * 0.5 * (np.log(1 - np.exp(-D * t)) + np.exp(-D * t)).sum(0)

