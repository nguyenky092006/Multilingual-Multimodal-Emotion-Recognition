"""Lightweight bottleneck residual adapters."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualAdapter(nn.Module):
    """LayerNorm → down projection → GELU → up projection with a residual path."""

    def __init__(self, d_model: int, bottleneck: int, dropout: float = 0.0) -> None:
        super().__init__()
        if bottleneck <= 0 or d_model <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.bias)

    def delta(self, hidden: Tensor) -> Tensor:
        return self.up(self.dropout(self.activation(self.down(self.norm(hidden)))))

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.delta(hidden)
