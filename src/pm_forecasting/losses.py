from __future__ import annotations

import torch
from torch.nn import functional as F


def normalized_to_raw(
    prediction: torch.Tensor,
    station: torch.Tensor,
    log_mean: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    mean = log_mean[station].unsqueeze(1)
    std = log_std[station].unsqueeze(1)
    return torch.expm1(prediction * std + mean).clamp_min(0)


def composite_loss(
    prediction: torch.Tensor,
    target_norm: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
    station: torch.Tensor,
    log_mean: torch.Tensor,
    log_std: torch.Tensor,
    raw_std: torch.Tensor,
    q90: torch.Tensor,
) -> torch.Tensor:
    """Proposal loss: log-Huber plus scaled absolute raw error, with peak weighting."""
    prediction_raw = normalized_to_raw(prediction, station, log_mean, log_std)
    huber = F.huber_loss(prediction, target_norm, delta=1.0, reduction="none")
    scale = raw_std[station].unsqueeze(1).clamp_min(1.0)
    raw_error = (prediction_raw - target_raw).abs() / scale
    peak_weight = torch.where(target_raw > q90[station].unsqueeze(1), 2.0, 1.0)
    element_loss = (huber + 0.5 * raw_error) * peak_weight
    valid = mask & torch.isfinite(target_raw)
    if not valid.any():
        return prediction.sum() * 0
    return element_loss[valid].mean()

