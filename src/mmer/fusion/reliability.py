"""Reliability-aware fusion with exact zero weights for unavailable branches."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mmer.data.cached import MODALITIES


def masked_softmax(scores: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    boolean_mask = mask.bool()
    if scores.shape != boolean_mask.shape:
        raise ValueError("scores and mask must have equal shape")
    if (~boolean_mask).all(dim=dim).any():
        raise ValueError("masked_softmax received an all-masked row")
    masked_scores = scores.masked_fill(~boolean_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights.masked_fill(~boolean_mask, 0.0)
    return weights / weights.sum(dim=dim, keepdim=True)


class ReliabilityGatedFusion(nn.Module):
    """Score only enabled modalities using representation, quality, and availability."""

    def __init__(
        self,
        d_model: int,
        hidden: int | None = None,
        dropout: float = 0.0,
        enabled_modalities: Sequence[str] = MODALITIES,
    ) -> None:
        super().__init__()
        requested = tuple(str(value) for value in enabled_modalities)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError(f"invalid enabled_modalities: {requested}")
        unknown = set(requested) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unsupported modalities: {sorted(unknown)}")
        self.enabled_modalities = tuple(name for name in MODALITIES if name in requested)
        gate_hidden = hidden or max(8, d_model // 2)
        self.scorers = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(d_model + 2),
                    nn.Linear(d_model + 2, gate_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden, 1),
                )
                for modality in self.enabled_modalities
            }
        )

    def forward(
        self,
        representations: Tensor,
        modality_mask: Tensor,
        quality: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if representations.ndim != 3 or representations.shape[1] != len(MODALITIES):
            raise ValueError("representations must have shape [batch, 3, d_model]")
        if quality.shape != modality_mask.shape or modality_mask.shape[1] != len(MODALITIES):
            raise ValueError("quality and modality_mask must have shape [batch, 3]")
        effective_mask = torch.zeros_like(modality_mask, dtype=torch.bool)
        score_columns: list[Tensor] = [
            torch.full(
                (representations.shape[0],),
                torch.finfo(representations.dtype).min,
                dtype=representations.dtype,
                device=representations.device,
            )
            for _ in MODALITIES
        ]
        for index, modality in enumerate(MODALITIES):
            if modality not in self.scorers:
                continue
            effective_mask[:, index] = modality_mask[:, index].bool()
            mask_float = effective_mask[:, index : index + 1].to(representations.dtype)
            quality_value = quality[:, index : index + 1].to(
                device=representations.device, dtype=representations.dtype
            )
            gate_input = torch.cat(
                [representations[:, index], quality_value, mask_float],
                dim=-1,
            )
            score_columns[index] = self.scorers[modality](gate_input).squeeze(-1)
        weights = masked_softmax(torch.stack(score_columns, dim=1), effective_mask, dim=1)
        fused = (representations * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights
