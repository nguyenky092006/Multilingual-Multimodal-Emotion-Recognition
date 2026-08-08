"""Lightweight language/corpus metadata conditioning baseline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class MetadataConditioner(nn.Module):
    """Add trainable language and corpus embeddings with explicit unknown routes."""

    def __init__(
        self,
        d_model: int,
        languages: Sequence[str],
        corpora: Sequence[str],
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("metadata alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self.language_ids = self._ids(languages)
        self.corpus_ids = self._ids(corpora)
        self.language_embedding = nn.Embedding(len(self.language_ids), d_model)
        self.corpus_embedding = nn.Embedding(len(self.corpus_ids), d_model)
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.language_embedding.weight, std=0.02)
        nn.init.normal_(self.corpus_embedding.weight, std=0.02)

    @staticmethod
    def _ids(values: Sequence[str]) -> dict[str, int]:
        unique = list(dict.fromkeys([*(str(value) for value in values), "unknown"]))
        return {value: index for index, value in enumerate(unique)}

    @staticmethod
    def _indices(values: Sequence[str], ids: dict[str, int], device: torch.device) -> Tensor:
        unknown = ids["unknown"]
        return torch.tensor([ids.get(str(value), unknown) for value in values], device=device)

    def forward(
        self,
        hidden: Tensor,
        languages: Sequence[str],
        corpora: Sequence[str],
    ) -> tuple[Tensor, dict[str, object]]:
        if hidden.ndim != 2 or len(languages) != hidden.shape[0] or len(corpora) != hidden.shape[0]:
            raise ValueError("metadata routes must match a [batch, d_model] tensor")
        language_indices = self._indices(languages, self.language_ids, hidden.device)
        corpus_indices = self._indices(corpora, self.corpus_ids, hidden.device)
        delta = self.language_embedding(language_indices) + self.corpus_embedding(corpus_indices)
        output = self.output_norm(hidden + self.alpha * delta)
        canonical_languages = [value if value in self.language_ids else "unknown" for value in languages]
        canonical_corpora = [value if value in self.corpus_ids else "unknown" for value in corpora]
        return output, {
            "enabled": True,
            "language_usage": dict(Counter(canonical_languages)),
            "corpus_usage": dict(Counter(canonical_corpora)),
            "output_norm": float(delta.detach().norm(dim=-1).mean().cpu()),
        }
