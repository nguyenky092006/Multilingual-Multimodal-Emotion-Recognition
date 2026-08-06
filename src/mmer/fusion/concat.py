"""Masked concatenation baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConcatenationFusion(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, representations: Tensor, modality_mask: Tensor) -> tuple[Tensor, Tensor]:
        if representations.ndim != 3 or representations.shape[1] != 3:
            raise ValueError("representations must have shape [batch, 3, d_model]")
        if (~modality_mask.bool()).all(dim=1).any():
            raise ValueError("fusion received an all-missing sample")
        masked = representations * modality_mask.unsqueeze(-1).to(representations.dtype)
        fused = self.network(masked.flatten(start_dim=1))
        weights = modality_mask.to(representations.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        return fused, weights
