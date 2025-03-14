import collections
import numbers
from torch_scatter import scatter_mean, scatter_sum
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from utils.mmcif import chi_pi_periodic_torch
from utils.residue_constants import blosum_numeric, blosum_62_cooccurrance_probs


def get_cooccur_score(res_pred, res_true, batch_idx):
    """
    Calculate co-occurrence score between predicted and true residue types.

    This function evaluates how well the predicted residues match the ground truth
    by using co-occurrence probabilities from the BLOSUM62 matrix. It accounts for
    similarity between amino acids, not just exact matches.

    Args:
        res_pred (torch.Tensor): Predicted residue indices
        res_true (torch.Tensor): Ground truth residue indices
        batch_idx (torch.Tensor): Batch indices for grouping results

    Returns:
        torch.Tensor: Normalized co-occurrence scores per batch element
    """
    probs = blosum_62_cooccurrance_probs.to(res_pred.device)
    return scatter_sum((probs[res_pred, res_true] + 4).float(), batch_idx, dim=-1) / scatter_sum((probs[res_true,res_true] + 4).float(), batch_idx, dim=-1)

def get_blosum_score(res_pred, res_true, batch_idx):
    """
    Calculate BLOSUM score between predicted and true residue types.

    This function computes a similarity score based on the BLOSUM substitution matrix,
    which captures evolutionary conservation patterns between amino acids.
    Higher scores indicate better prediction quality.

    Args:
        res_pred (torch.Tensor): Predicted residue indices
        res_true (torch.Tensor): Ground truth residue indices
        batch_idx (torch.Tensor): Batch indices for grouping results

    Returns:
        torch.Tensor: Normalized BLOSUM scores per batch element
    """
    blosum = blosum_numeric.to(res_pred.device)
    return scatter_sum((blosum[res_pred, res_true] + 4).float(), batch_idx, dim=-1) / scatter_sum((blosum[res_true,res_true] + 4).float(), batch_idx, dim=-1)

def get_unnorm_blosum_score(res_pred, res_true, batch_idx):
    """
    Calculate unnormalized BLOSUM score between predicted and true residue types.

    Similar to get_blosum_score but without adding the +4 offset that shifts
    all values to be positive. This version maintains the original substitution
    values which can be negative for dissimilar amino acids.

    Args:
        res_pred (torch.Tensor): Predicted residue indices
        res_true (torch.Tensor): Ground truth residue indices
        batch_idx (torch.Tensor): Batch indices for grouping results

    Returns:
        torch.Tensor: Unnormalized BLOSUM scores per batch element
    """
    blosum = blosum_numeric.to(res_pred.device)
    return scatter_sum((blosum[res_pred, res_true]).float(), batch_idx, dim=-1) / scatter_sum((blosum[res_true,res_true]).float(), batch_idx, dim=-1)

def compute_rmsds(true_pos, x0, batch):
    """
    Compute various root-mean-square deviation (RMSD) metrics between predicted and true ligand positions.

    This function calculates three types of RMSD to evaluate structural prediction quality:
    1. Standard RMSD between predicted and true atom positions
    2. Centroid RMSD (distance between centers of mass)
    3. Kabsch RMSD (minimum RMSD after optimal alignment)

    Args:
        true_pos (torch.Tensor): Ground truth atom positions
        x0 (torch.Tensor): Predicted atom positions
        batch (object): Batch data containing graph information

    Returns:
        tuple: (rmsd, cent_rmsd, kabsch_rmsd) arrays with values for each molecule in the batch
    """
    rmsd = scatter_mean(torch.square(true_pos - x0).sum(-1), batch['ligand'].batch) ** 0.5
    centroid = scatter_mean(x0, batch['ligand'].batch, 0)
    true_cent = scatter_mean(true_pos, batch['ligand'].batch, 0)
    cent_rmsd = torch.square(centroid - true_cent).sum(-1) ** 0.5

    kabsch_rmsd = []
    for i in range(batch.num_graphs):
        x0_ = x0[batch['ligand'].batch == i].cpu().numpy()
        true_pos_ = true_pos[batch['ligand'].batch == i].cpu().numpy()
        try:
            kabsch_rmsd.append(
                Rotation.align_vectors(x0_, true_pos_)[1] / np.sqrt(x0_.shape[0])
            )
        except:
            kabsch_rmsd.append(np.inf)
    return rmsd.cpu().numpy(), cent_rmsd.cpu().numpy(), np.array(kabsch_rmsd)

def squared_difference(x, y):
    """
    Compute element-wise squared difference between two arrays.

    A simple utility function to calculate (x - y)² for loss functions.

    Args:
        x (torch.Tensor): First tensor
        y (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: Element-wise squared differences
    """
    return torch.square(x - y)


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    """
    Compute a masked mean along specified axes.

    This function calculates the mean of 'value' elements where 'mask' is non-zero,
    handling various broadcast scenarios and avoiding division by zero.

    Args:
        mask (torch.Tensor): Binary mask indicating which values to include
        value (torch.Tensor): Values to average
        axis (int or list, optional): Dimensions to reduce. Defaults to all dimensions.
        drop_mask_channel (bool, optional): Whether to drop last dimension of mask. Defaults to False.
        eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-10.

    Returns:
        torch.Tensor: Masked mean values
    """
    if drop_mask_channel:
        mask = mask[..., 0]

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    assert isinstance(axis, collections.abc.Iterable), 'axis needs to be either an iterable, integer or "None"'

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size

    return torch.sum(mask * value, dim=axis) / (torch.sum(mask, dim=axis) * broadcast_factor + eps)

def angle_unit_loss(preds, mask, eps=1e-6):
    """
    Calculate loss to enforce unit length for angle vectors.

    For angle predictions in sin/cos form, this function penalizes deviations from
    unit norm to ensure valid angles (sin²θ + cos²θ = 1).

    Args:
        preds (torch.Tensor): Predicted angle vectors
        mask (torch.Tensor): Binary mask for valid entries
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        torch.Tensor: Loss value measuring deviation from unit norm
    """
    # Aux loss to keep vectors in unit circle
    angle_norm = torch.sqrt(torch.sum(torch.square(preds), dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.)
    angle_norm_loss = mask_mean(mask=mask[..., None], value=norm_error)
    return angle_norm_loss

def l2_normalize(x, axis=-1, epsilon=1e-12):
    """
    Normalize vectors to unit L2 norm along specified axis.

    Ensures that vectors have length 1 by dividing by their magnitude.
    Includes protection against division by zero.

    Args:
        x (torch.Tensor): Input tensor to normalize
        axis (int, optional): Axis along which to normalize. Defaults to -1.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-12.

    Returns:
        torch.Tensor: Normalized tensor with unit norm along specified axis
    """
    y = torch.sum(x**2, dim=axis, keepdim=True)
    return x / torch.sqrt(torch.maximum(y, torch.ones_like(y) * epsilon))

def supervised_chi_loss(batch, preds, angles_idx_s=0, angles_idx=11):
    """
    Calculate supervised loss for side chain torsion angle prediction.

    This function computes the error between predicted and ground truth torsion angles,
    handling their periodic nature and potential symmetric rotations.

    Args:
        batch (object): Batch data containing ground truth angles and masks
        preds (torch.Tensor): Predicted angles in sin/cos encoding
        angles_idx_s (int, optional): Start index for angles to consider. Defaults to 0.
        angles_idx (int, optional): End index for angles to consider. Defaults to 11.

    Returns:
        torch.Tensor: Combined loss value for angle prediction
    """
    chi_mask = batch['protein'].angle_mask[:, angles_idx_s:angles_idx]
    sin_cos_true_chi = batch["protein"].angles[:, angles_idx_s:angles_idx, :]  # 3 torsion + 3 angle + 5 side chain torsion = 11

    # Extend to backbone angle / torsion angles besides side chain chi angles
    # TODO move somewhere else, inefficient to redefine each time
    chi_pi_periodic = chi_pi_periodic_torch.to(preds.device)[:, angles_idx_s:angles_idx]

    # L2 normalized predicted angles
    angles_sin_cos = l2_normalize(preds, axis=-1)  # [:, :, angles_idx_s:angles_idx, :]

    # One-hot encode and apply periodic mask
    chi_pi_periodic = chi_pi_periodic[batch['protein'].aatype_num]

    # This is -1 if chi is pi-periodic and +1 if it's 2pi-periodic
    shifted_mask = (1 - 2 * chi_pi_periodic)[..., None]
    sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi  # Add + pi if rotation-symmetric

    # Main torsion loss
    sq_chi_error = torch.sum(squared_difference(sin_cos_true_chi, angles_sin_cos), -1)
    sq_chi_error_shifted = torch.sum(squared_difference(sin_cos_true_chi_shifted, angles_sin_cos), -1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    sq_chi_loss = mask_mean(mask=chi_mask, value=sq_chi_error)

    # Aux loss to keep vectors in unit circle
    angle_norm_loss = angle_unit_loss(preds, batch['protein'].is_canonical)

    # Final loss
    loss = sq_chi_loss + 0.02 * angle_norm_loss

    return loss

def get_recovered_aa_angle_loss(batch, angles, res_pred, angles_idx_s=0, angles_idx=11):
    """
    Calculate torsion angle loss for correctly predicted amino acids only.

    This function first identifies residues that were correctly predicted,
    then calculates the angle prediction loss only for those positions.
    This provides a more focused evaluation of side chain conformation quality.

    Args:
        batch (object): Batch data containing ground truth information
        angles (torch.Tensor): Predicted angles in sin/cos encoding
        res_pred (torch.Tensor): Predicted residue type logits
        angles_idx_s (int, optional): Start index for angles to consider. Defaults to 0.
        angles_idx (int, optional): End index for angles to consider. Defaults to 11.

    Returns:
        torch.Tensor: Angle prediction loss for correctly predicted residues
    """
    if angles is None:
        return torch.tensor(0.0)
    correctly_predicted = torch.zeros_like(batch['protein'].designable_mask).bool()
    correctly_predicted[torch.where(torch.argmax(res_pred, dim=1) == batch['protein'].feat[:, 0].view(-1))[0]] = True
    batch["protein"].angles = batch["protein"].angles[correctly_predicted]
    batch['protein'].angle_mask = batch['protein'].angle_mask[correctly_predicted]
    batch['protein'].aatype_num = batch['protein'].aatype_num[correctly_predicted]
    batch['protein'].is_canonical = batch['protein'].is_canonical[correctly_predicted]
    angles = angles[correctly_predicted]
    if correctly_predicted.sum() == 0:
        return torch.tensor(0.0)
    return supervised_chi_loss(batch, angles, angles_idx_s=angles_idx_s, angles_idx=angles_idx)
def energy_prediction_loss(predicted_energy, target_energy, batch_idx=None):
    """
    Calculate mean squared error loss for energy prediction.

    Args:
        predicted_energy (torch.Tensor): Predicted binding energy values
        target_energy (torch.Tensor): Ground truth binding energy values
        batch_idx (torch.Tensor, optional): Batch indices for scatter operations

    Returns:
        torch.Tensor: MSE loss for energy prediction
    """
    # Standard MSE loss
    loss = torch.nn.functional.mse_loss(predicted_energy, target_energy, reduction='none')

    # If batch indices are provided, compute per-graph average
    if batch_idx is not None:
        return scatter_mean(loss, batch_idx, dim=-1)
    return loss.mean()

def confidence_score_loss(pred_confidence_score, batch, batch_idx=None, scale_factor=5.0):
    """
    Calculate loss for confidence prediction based on RMSD values.
    Higher RMSD should correspond to lower confidence.

    Args:
        pred_confidence_score (torch.Tensor): Predicted confidence scores [0,1]
        batch (torch.Tensor): Batch input
        batch_idx (torch.Tensor, optional): Batch indices for scatter operations
        scale_factor (float): Factor to scale RMSD values for mapping to [0,1] range

    Returns:
        Tuple[torch.Tensor]: Loss for confidence prediction and confidence correlation
    """
    # Calculate RMSD between predicted and actual positions
    # as ground truth for confidence (lower RMSD = higher confidence)
    squared_diff = torch.square(batch['ligand'].shadow_pos - batch['ligand'].pos).sum(-1)
    rmsd_per_atom = torch.sqrt(squared_diff)
    rmsd_per_graph = scatter_mean(rmsd_per_atom, batch_idx, -1)

    # Calculate confidence loss using RMSD-based target
    # Lower RMSD should correspond to higher confidence
    scale_factor = 5.0  # Adjust this to calibrate the confidence scale
    target_confidence = torch.exp(-rmsd_per_graph / scale_factor)

    confidence_loss_raw = torch.nn.functional.binary_cross_entropy(
        pred_confidence_score,
        target_confidence,
        reduction='none'
    )

    if batch_idx is not None:
        confidence_loss = scatter_mean(confidence_loss_raw, torch.arange(len(pred_confidence_score), device=pred_confidence_score.device), -1)
    else:
        confidence_loss = confidence_loss_raw.mean()

    # Calculate correlation between confidence and actual RMSD
    # Higher confidence should correlate with lower RMSD
    # Here we use a simple inverse relationship metric
    confidence_corr = scatter_mean(
        (1.0 - pred_confidence_score) * rmsd_per_graph,
        torch.arange(len(pred_confidence_score), device=pred_confidence_score.device),
        -1
    )

    return confidence_loss, confidence_corr