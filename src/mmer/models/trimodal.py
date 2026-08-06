"""Trimodal classifier with shared and routed lightweight adapters."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from mmer.adapters import AdapterRouter, ResidualAdapter
from mmer.data.cached import MODALITIES
from mmer.fusion import ConcatenationFusion, ReliabilityGatedFusion
from .projections import AudioProjection, TextProjection, VisualProjection


class TrimodalEmotionModel(nn.Module):
    """Trainable head operating only on frozen/cached encoder representations."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        num_classes: int,
        languages: Sequence[str],
        corpora: Sequence[str],
        d_model: int = 256,
        projection_hidden: int = 512,
        adapter_bottleneck: int = 64,
        dropout: float = 0.2,
        fusion: str = "reliability",
        shared_adapter: bool = True,
        routing_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        missing_dims = set(MODALITIES) - set(input_dims)
        if missing_dims:
            raise ValueError(f"missing input dimensions: {sorted(missing_dims)}")
        projection_types = (AudioProjection, TextProjection, VisualProjection)
        self.projections = nn.ModuleDict(
            {
                modality: projection_type(int(input_dims[modality]), d_model, projection_hidden, dropout)
                for modality, projection_type in zip(MODALITIES, projection_types, strict=True)
            }
        )
        self.modality_adapters = nn.ModuleDict(
            {modality: ResidualAdapter(d_model, adapter_bottleneck, dropout) for modality in MODALITIES}
        )
        self.shared_adapter_enabled = shared_adapter
        if shared_adapter:
            self.emotion_adapter: nn.Module = ResidualAdapter(d_model, adapter_bottleneck, dropout)
        else:
            self.emotion_adapter = nn.ModuleDict(
                {modality: ResidualAdapter(d_model, adapter_bottleneck, dropout) for modality in MODALITIES}
            )
        self.router = AdapterRouter(
            d_model, adapter_bottleneck, languages, corpora, routing_alpha, dropout
        )
        if fusion == "reliability":
            self.fusion = ReliabilityGatedFusion(d_model, dropout=dropout)
        elif fusion == "concat":
            self.fusion = ConcatenationFusion(d_model, dropout)
        else:
            raise ValueError(f"unsupported fusion: {fusion}")
        self.fusion_name = fusion
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        embeddings: Mapping[str, Tensor],
        modality_mask: Tensor,
        languages: Sequence[str],
        corpora: Sequence[str],
        quality: Tensor | None = None,
    ) -> dict[str, object]:
        if modality_mask.ndim != 2 or modality_mask.shape[1] != 3:
            raise ValueError("modality_mask must have shape [batch, 3]")
        if (~modality_mask.bool()).all(dim=1).any():
            raise ValueError("model received an all-missing sample")
        representations: list[Tensor] = []
        route_stats: dict[str, object] = {}
        for index, modality in enumerate(MODALITIES):
            hidden = self.projections[modality](embeddings[modality])
            hidden = self.modality_adapters[modality](hidden)
            if self.shared_adapter_enabled:
                hidden = self.emotion_adapter(hidden)
            else:
                hidden = self.emotion_adapter[modality](hidden)  # type: ignore[index]
            hidden, stats = self.router(hidden, languages, corpora)
            hidden = hidden * modality_mask[:, index : index + 1].to(hidden.dtype)
            representations.append(hidden)
            route_stats[modality] = stats
        stacked = torch.stack(representations, dim=1)
        if self.fusion_name == "reliability":
            if quality is None:
                raise ValueError("reliability fusion requires quality features")
            fused, fusion_weights = self.fusion(stacked, modality_mask, quality)
        else:
            fused, fusion_weights = self.fusion(stacked, modality_mask)
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "representations": stacked,
            "fusion_weights": fusion_weights,
            "route_stats": route_stats,
        }

    @staticmethod
    def summarise_fusion_weights(
        weights: Tensor,
        languages: Sequence[str],
        corpora: Sequence[str],
        emotions: Sequence[str],
        modality_mask: Tensor,
    ) -> dict[str, dict[str, list[float]]]:
        result: dict[str, dict[str, list[float]]] = {}
        dimensions = {
            "language": languages,
            "corpus": corpora,
            "emotion": emotions,
            "modality_pattern": ["".join("ATV"[i] for i, flag in enumerate(row) if flag) for row in modality_mask.tolist()],
        }
        detached = weights.detach().cpu()
        for dimension, values in dimensions.items():
            buckets: dict[str, list[Tensor]] = defaultdict(list)
            for index, value in enumerate(values):
                buckets[str(value)].append(detached[index])
            result[dimension] = {
                key: torch.stack(rows).mean(dim=0).tolist() for key, rows in sorted(buckets.items())
            }
        return result


def parameter_counts(module: nn.Module) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }

