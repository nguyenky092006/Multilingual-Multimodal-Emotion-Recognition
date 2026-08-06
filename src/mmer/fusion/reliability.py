"""Reliability-aware fusion with exact zero weights for missing modalities."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_softmax(scores: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    boolean_mask = mask.bool()
    if (~boolean_mask).all(dim=dim).any():
        raise ValueError("masked_softmax received an all-masked row")
    masked_scores = scores.masked_fill(~boolean_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights.masked_fill(~boolean_mask, 0.0)
    return weights / weights.sum(dim=dim, keepdim=True)


class ReliabilityGatedFusion(nn.Module):
    """Score each modality using its representation, quality, and availability."""

    def __init__(self, d_model: int, hidden: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        gate_hidden = hidden or max(8, d_model // 2)
        self.scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model + 2),
                    nn.Linear(d_model + 2, gate_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden, 1),
                )
                for _ in range(3)
            ]
        )

    def forward(
        self,
        representations: Tensor,
        modality_mask: Tensor,
        quality: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if representations.ndim != 3 or representations.shape[1] != 3:
            raise ValueError("representations must have shape [batch, 3, d_model]")
        if quality.shape != modality_mask.shape or modality_mask.shape[1] != 3:
            raise ValueError("quality and modality_mask must have shape [batch, 3]")
        mask_float = modality_mask.to(representations.dtype)
        scores = []
        for index, scorer in enumerate(self.scorers):
            gate_input = torch.cat(
                [representations[:, index], quality[:, index : index + 1], mask_float[:, index : index + 1]],
                dim=-1,
            )
            scores.append(scorer(gate_input).squeeze(-1))
        weights = masked_softmax(torch.stack(scores, dim=1), modality_mask, dim=1)
        fused = (representations * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights

