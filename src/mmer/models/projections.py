"""Trainable modality projections into a shared representation space."""

from __future__ import annotations

from torch import nn


class ModalityProjection(nn.Sequential):
    def __init__(self, input_dim: int, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )


class AudioProjection(ModalityProjection):
    pass


class TextProjection(ModalityProjection):
    pass


class VisualProjection(ModalityProjection):
    pass

