"""Batched language- and corpus-specific residual adapter routing."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import torch
from torch import Tensor, nn

from .residual import ResidualAdapter


class AdapterRouter(nn.Module):
    """Route each batch row through a language and corpus residual adapter."""

    def __init__(
        self,
        d_model: int,
        bottleneck: int,
        languages: Sequence[str],
        corpora: Sequence[str],
        alpha: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.language_keys = self._route_keys(languages)
        self.corpus_keys = self._route_keys(corpora)
        self.language_adapters = nn.ModuleDict(
            {key: ResidualAdapter(d_model, bottleneck, dropout) for key in self.language_keys.values()}
        )
        self.corpus_adapters = nn.ModuleDict(
            {key: ResidualAdapter(d_model, bottleneck, dropout) for key in self.corpus_keys.values()}
        )
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _route_keys(values: Sequence[str]) -> dict[str, str]:
        unique = list(dict.fromkeys([*values, "unknown"]))
        return {value: f"route_{index}" for index, value in enumerate(unique)}

    @staticmethod
    def _apply_grouped(
        hidden: Tensor,
        routes: Sequence[str],
        keys: dict[str, str],
        adapters: nn.ModuleDict,
    ) -> tuple[Tensor, dict[str, int], dict[str, float]]:
        if len(routes) != hidden.shape[0]:
            raise ValueError("route count must equal batch size")
        canonical = [route if route in keys else "unknown" for route in routes]
        result = torch.zeros_like(hidden)
        usage = Counter(canonical)
        norms: dict[str, list[float]] = defaultdict(list)
        for route in dict.fromkeys(canonical):
            indices = torch.tensor(
                [i for i, value in enumerate(canonical) if value == route],
                device=hidden.device,
                dtype=torch.long,
            )
            delta = adapters[keys[route]].delta(hidden.index_select(0, indices))
            result.index_copy_(0, indices, delta)
            norms[route].append(float(delta.detach().norm(dim=-1).mean().cpu()))
        return result, dict(usage), {key: sum(values) / len(values) for key, values in norms.items()}

    def forward(
        self,
        hidden: Tensor,
        languages: Sequence[str],
        corpora: Sequence[str],
    ) -> tuple[Tensor, dict[str, object]]:
        language_delta, language_usage, language_norms = self._apply_grouped(
            hidden, languages, self.language_keys, self.language_adapters
        )
        corpus_delta, corpus_usage, corpus_norms = self._apply_grouped(
            hidden, corpora, self.corpus_keys, self.corpus_adapters
        )
        output = self.output_norm(hidden + self.alpha * (language_delta + corpus_delta))
        all_norms = [*language_norms.values(), *corpus_norms.values()]
        stats: dict[str, object] = {
            "language_usage": language_usage,
            "corpus_usage": corpus_usage,
            "language_output_norm": language_norms,
            "corpus_output_norm": corpus_norms,
            "collapse_detected": bool(all_norms and max(all_norms) < 1e-8),
        }
        return output, stats
