"""Masked temporal pooling for cached frame-level visual representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate(features: Tensor, mask: Tensor) -> None:
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError("temporal features/mask must have shapes [batch,time,dim]/[batch,time]")
    if (~mask.bool()).all(dim=1).any():
        raise ValueError("temporal mask contains an all-padding sequence")


class MaskedTemporalMean(nn.Module):
    """Parameter-free masked mean over cached frame representations."""

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        _validate(features, mask)
        weights = mask.to(features.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        return (features * weights.unsqueeze(-1)).sum(dim=1), weights


class TemporalAttentionPool(nn.Module):
    """Small trainable attention scorer over projected frame embeddings."""

    def __init__(self, d_model: int, hidden: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        attention_hidden = int(hidden or max(8, d_model // 2))
        if d_model <= 0 or attention_hidden <= 0:
            raise ValueError("temporal attention dimensions must be positive")
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, attention_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden, 1),
        )

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        _validate(features, mask)
        logits = self.scorer(features).squeeze(-1)
        minimum = torch.finfo(logits.dtype).min
        weights = torch.softmax(logits.masked_fill(~mask.bool(), minimum), dim=1)
        weights = weights.masked_fill(~mask.bool(), 0.0)
        return (features * weights.unsqueeze(-1)).sum(dim=1), weights
