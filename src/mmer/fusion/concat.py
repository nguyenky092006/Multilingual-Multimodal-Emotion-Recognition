"""Masked concatenation for any configured subset of audio, text, and visual."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mmer.data.cached import MODALITIES


def _active_indices(enabled_modalities: Sequence[str]) -> tuple[int, ...]:
    requested = tuple(str(value) for value in enabled_modalities)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError(f"invalid enabled_modalities: {requested}")
    unknown = set(requested) - set(MODALITIES)
    if unknown:
        raise ValueError(f"unsupported modalities: {sorted(unknown)}")
    return tuple(index for index, name in enumerate(MODALITIES) if name in requested)


class ConcatenationFusion(nn.Module):
    """Concatenate only active branches while returning canonical three-way weights."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        enabled_modalities: Sequence[str] = MODALITIES,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.active_indices = _active_indices(enabled_modalities)
        self.network = nn.Sequential(
            nn.LayerNorm(len(self.active_indices) * d_model),
            nn.Linear(len(self.active_indices) * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, representations: Tensor, modality_mask: Tensor) -> tuple[Tensor, Tensor]:
        if representations.ndim != 3 or representations.shape[1] != len(MODALITIES):
            raise ValueError("representations must have shape [batch, 3, d_model]")
        if modality_mask.shape != representations.shape[:2]:
            raise ValueError("modality_mask must have shape [batch, 3]")
        active_mask = torch.zeros_like(modality_mask, dtype=torch.bool)
        active_mask[:, self.active_indices] = modality_mask[:, self.active_indices].bool()
        if (~active_mask).all(dim=1).any():
            raise ValueError("fusion received an all-missing sample")
        selected = representations[:, self.active_indices]
        selected_mask = active_mask[:, self.active_indices]
        masked = selected * selected_mask.unsqueeze(-1).to(representations.dtype)
        fused = self.network(masked.flatten(start_dim=1))
        weights = active_mask.to(representations.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        return fused, weights
