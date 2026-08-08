"""Optional losses for the full meta-adapter experiment."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def supervised_contrastive_loss(
    features: Tensor,
    labels: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Supervised contrastive loss over a single episode's shared embeddings."""

    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have shapes [samples, dim] and [samples]")
    if features.shape[0] < 2 or temperature <= 0:
        raise ValueError("contrastive loss needs at least two samples and positive temperature")
    normalized = F.normalize(features, dim=-1)
    logits = normalized @ normalized.T / temperature
    diagonal = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positive.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(diagonal, 0.0)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = (log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return -mean_positive[valid].mean()
