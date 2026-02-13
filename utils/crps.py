import torch
import torch.nn.functional as F

def crps_from_samples(pred_samples, true_values, dim=0, sorted=False):
    """
    Computes the Continuous Ranked Probability Score (CRPS) from ensemble samples.
    Reference: Gneiting et al. (2005) "Calibrated Probabilistic Forecasting Using Ensemble Methods"

    Args:
        pred_samples (Tensor): Ensemble predictions of shape [S, N, T] 
                               (S: num_samples, N: batch_size, T: time_steps).
        true_values (Tensor): Ground truth values of shape [N, T].
        dim (int): The dimension representing the ensemble samples. Default: 0.
        sorted (bool): If True, pred_samples is assumed to be sorted along dim.

    Returns:
        Tensor: CRPS values for each instance, shape [N, T].
    """
    if pred_samples.dim() != 3:
        pred_samples = pred_samples.squeeze(-1)
        true_values = true_values.squeeze(-1)
        
    S, N, T = pred_samples.shape

    if not sorted:
        pred_samples, _ = torch.sort(pred_samples, dim=dim)

    indices = torch.arange(1, S + 1, device=pred_samples.device, dtype=pred_samples.dtype)
    weights = (2 * indices - 1) / (S ** 2)
    weights = weights.view(S, 1, 1)  

    true_values_expanded = true_values.unsqueeze(dim=dim)
    term1 = torch.sum(weights * (pred_samples - true_values_expanded), dim=dim)

    cum_sum = torch.cumsum(pred_samples, dim=dim)
    indices_3d = indices.view(S, 1, 1)
    term2 = (2 / (S ** 2)) * torch.sum(indices_3d * pred_samples - cum_sum, dim=dim)

    crps = term1 - term2
    return torch.clamp(crps, min=0.0)
