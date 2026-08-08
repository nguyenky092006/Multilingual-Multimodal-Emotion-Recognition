"""Configurable multimodal classifier with lightweight routed adapters."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from mmer.adapters import AdapterRouter, MetadataConditioner, ResidualAdapter
from mmer.data.cached import MODALITIES
from mmer.fusion import ConcatenationFusion, ReliabilityGatedFusion
from .projections import AudioProjection, TextProjection, VisualProjection
from .temporal import MaskedTemporalMean, TemporalAttentionPool


class TrimodalEmotionModel(nn.Module):
    """Trainable head operating only on frozen/cached encoder representations.

    The output contract always uses canonical audio/text/visual columns, but modules and
    optimizer parameters are created only for ``enabled_modalities``. This makes a true
    audio-text or unimodal run independent from disabled cache branches.
    """

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
        enabled_modalities: Sequence[str] = MODALITIES,
        use_modality_adapters: bool = True,
        emotion_adapter_mode: str | None = None,
        use_routed_adapters: bool = True,
        use_metadata_embeddings: bool = False,
        metadata_alpha: float = 0.5,
        modality_adapter_bottleneck: int | None = None,
        emotion_adapter_bottleneck: int | None = None,
        routed_adapter_bottleneck: int | None = None,
        visual_temporal_pooling: str = "prepooled",
        temporal_attention_hidden: int | None = None,
        use_language_adapters: bool = True,
        use_corpus_adapters: bool = True,
    ) -> None:
        super().__init__()
        requested = tuple(str(value) for value in enabled_modalities)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError(f"invalid enabled_modalities: {requested}")
        unknown = set(requested) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unsupported modalities: {sorted(unknown)}")
        self.enabled_modalities = tuple(name for name in MODALITIES if name in requested)
        missing_dims = set(self.enabled_modalities) - set(input_dims)
        if missing_dims:
            raise ValueError(f"missing input dimensions: {sorted(missing_dims)}")
        if any(int(input_dims[name]) <= 0 for name in self.enabled_modalities):
            raise ValueError("enabled input dimensions must be positive")
        if num_classes <= 0 or d_model <= 0 or projection_hidden <= 0:
            raise ValueError("model dimensions and num_classes must be positive")
        if not 0.0 <= float(routing_alpha) <= 1.0:
            raise ValueError("routing_alpha must be in [0, 1]")
        modality_bottleneck = int(
            adapter_bottleneck
            if modality_adapter_bottleneck is None
            else modality_adapter_bottleneck
        )
        emotion_bottleneck = int(
            adapter_bottleneck
            if emotion_adapter_bottleneck is None
            else emotion_adapter_bottleneck
        )
        routed_bottleneck = int(
            adapter_bottleneck
            if routed_adapter_bottleneck is None
            else routed_adapter_bottleneck
        )
        if min(modality_bottleneck, emotion_bottleneck, routed_bottleneck) <= 0:
            raise ValueError("all adapter bottlenecks must be positive")
        if visual_temporal_pooling not in {"prepooled", "mean", "attention"}:
            raise ValueError("visual_temporal_pooling must be prepooled, mean, or attention")

        projection_types = {
            "audio": AudioProjection,
            "text": TextProjection,
            "visual": VisualProjection,
        }
        self.d_model = int(d_model)
        self.projections = nn.ModuleDict(
            {
                modality: projection_types[modality](
                    int(input_dims[modality]), d_model, projection_hidden, dropout
                )
                for modality in self.enabled_modalities
            }
        )
        self.modality_adapters = nn.ModuleDict(
            {
                modality: (
                    ResidualAdapter(d_model, modality_bottleneck, dropout)
                    if use_modality_adapters
                    else nn.Identity()
                )
                for modality in self.enabled_modalities
            }
        )

        mode = emotion_adapter_mode or ("shared" if shared_adapter else "separate")
        if mode not in {"shared", "separate", "none"}:
            raise ValueError("emotion_adapter_mode must be shared, separate, or none")
        self.emotion_adapter_mode = mode
        if mode == "shared":
            self.emotion_adapter: nn.Module = ResidualAdapter(
                d_model, emotion_bottleneck, dropout
            )
        elif mode == "separate":
            self.emotion_adapter = nn.ModuleDict(
                {
                    modality: ResidualAdapter(d_model, emotion_bottleneck, dropout)
                    for modality in self.enabled_modalities
                }
            )
        else:
            self.emotion_adapter = nn.Identity()

        self.router = (
            AdapterRouter(
                d_model,
                routed_bottleneck,
                languages,
                corpora,
                routing_alpha,
                dropout,
                use_language_adapters,
                use_corpus_adapters,
            )
            if use_routed_adapters
            else None
        )
        self.visual_temporal_pooling = visual_temporal_pooling
        if visual_temporal_pooling == "mean":
            self.visual_temporal_pool: nn.Module | None = MaskedTemporalMean()
        elif visual_temporal_pooling == "attention":
            self.visual_temporal_pool = TemporalAttentionPool(
                d_model, temporal_attention_hidden, dropout
            )
        else:
            self.visual_temporal_pool = None
        self.metadata_conditioner = (
            MetadataConditioner(d_model, languages, corpora, metadata_alpha)
            if use_metadata_embeddings
            else None
        )
        if fusion == "reliability":
            self.fusion = ReliabilityGatedFusion(
                d_model, dropout=dropout, enabled_modalities=self.enabled_modalities
            )
        elif fusion == "concat":
            self.fusion = ConcatenationFusion(
                d_model, dropout, enabled_modalities=self.enabled_modalities
            )
        else:
            raise ValueError(f"unsupported fusion: {fusion}")
        self.fusion_name = fusion
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _emotion_adapt(self, modality: str, hidden: Tensor) -> Tensor:
        if self.emotion_adapter_mode == "shared":
            return self.emotion_adapter(hidden)
        if self.emotion_adapter_mode == "separate":
            return self.emotion_adapter[modality](hidden)  # type: ignore[index]
        return hidden

    def forward(
        self,
        embeddings: Mapping[str, Tensor],
        modality_mask: Tensor,
        languages: Sequence[str],
        corpora: Sequence[str],
        quality: Tensor | None = None,
        temporal_masks: Mapping[str, Tensor] | None = None,
    ) -> dict[str, object]:
        if modality_mask.ndim != 2 or modality_mask.shape[1] != len(MODALITIES):
            raise ValueError("modality_mask must have shape [batch, 3]")
        batch_size = modality_mask.shape[0]
        effective_mask = torch.zeros_like(modality_mask, dtype=torch.bool)
        active_indices = [MODALITIES.index(name) for name in self.enabled_modalities]
        effective_mask[:, active_indices] = modality_mask[:, active_indices].bool()
        if (~effective_mask).all(dim=1).any():
            raise ValueError("model received an all-missing sample after applying enabled_modalities")
        if len(languages) != batch_size or len(corpora) != batch_size:
            raise ValueError("language and corpus counts must equal batch size")

        reference: Tensor | None = None
        active_outputs: dict[str, Tensor] = {}
        shared_outputs: dict[str, Tensor] = {}
        route_stats: dict[str, object] = {}
        temporal_weights: dict[str, Tensor] = {}
        for modality in self.enabled_modalities:
            if modality not in embeddings:
                raise ValueError(f"missing enabled embedding: {modality}")
            raw = embeddings[modality]
            if raw.shape[0] != batch_size or raw.ndim not in {2, 3}:
                raise ValueError(
                    f"{modality} embedding must have shape [batch,features] or "
                    "[batch,time,features]"
                )
            if raw.ndim == 3:
                if modality != "visual" or self.visual_temporal_pool is None:
                    raise ValueError(
                        f"{modality} temporal embeddings require configured temporal pooling"
                    )
                if temporal_masks is None or modality not in temporal_masks:
                    raise ValueError(f"missing temporal mask for {modality}")
                hidden_frames = self.projections[modality](raw)
                hidden, attention = self.visual_temporal_pool(
                    hidden_frames, temporal_masks[modality]
                )
                temporal_weights[modality] = attention
            else:
                if modality == "visual" and self.visual_temporal_pooling != "prepooled":
                    raise ValueError(
                        "visual temporal pooling requires frame-level cached embeddings"
                    )
                hidden = self.projections[modality](raw)
            hidden = self.modality_adapters[modality](hidden)
            hidden = self._emotion_adapt(modality, hidden)
            shared_outputs[modality] = hidden
            metadata_stats: dict[str, object] = {"enabled": False}
            if self.metadata_conditioner is not None:
                hidden, metadata_stats = self.metadata_conditioner(hidden, languages, corpora)
            adapter_stats: dict[str, object] = {"enabled": False}
            if self.router is not None:
                hidden, adapter_stats = self.router(hidden, languages, corpora)
                adapter_stats = {"enabled": True, **adapter_stats}
            route_stats[modality] = {
                "metadata": metadata_stats,
                "adapter": adapter_stats,
            }
            index = MODALITIES.index(modality)
            hidden = hidden * effective_mask[:, index : index + 1].to(hidden.dtype)
            active_outputs[modality] = hidden
            reference = hidden
        assert reference is not None

        representations = [
            active_outputs.get(modality, torch.zeros_like(reference)) for modality in MODALITIES
        ]
        shared_representations = [
            shared_outputs.get(modality, torch.zeros_like(reference)) for modality in MODALITIES
        ]
        stacked = torch.stack(representations, dim=1)
        stacked_shared = torch.stack(shared_representations, dim=1)
        if self.fusion_name == "reliability":
            if quality is None:
                raise ValueError("reliability fusion requires quality features")
            if quality.shape != effective_mask.shape:
                raise ValueError("quality must have shape [batch, 3]")
            fused, fusion_weights = self.fusion(stacked, effective_mask, quality)
        else:
            fused, fusion_weights = self.fusion(stacked, effective_mask)
        shared_weights = effective_mask.to(stacked_shared.dtype)
        shared_weights = shared_weights / shared_weights.sum(dim=1, keepdim=True)
        shared_fused = (stacked_shared * shared_weights.unsqueeze(-1)).sum(dim=1)
        return {
            "logits": self.classifier(fused),
            "fused": fused,
            "representations": stacked,
            "shared_representations": stacked_shared,
            "shared_fused": shared_fused,
            "shared_fusion_weights": shared_weights,
            "temporal_weights": temporal_weights,
            "fusion_weights": fusion_weights,
            "route_stats": route_stats,
            "effective_modality_mask": effective_mask,
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
            "modality_pattern": [
                "".join("ATV"[i] for i, flag in enumerate(row) if flag)
                for row in modality_mask.tolist()
            ],
        }
        detached = weights.detach().cpu()
        for dimension, values in dimensions.items():
            buckets: dict[str, list[Tensor]] = defaultdict(list)
            for index, value in enumerate(values):
                buckets[str(value)].append(detached[index])
            result[dimension] = {
                key: torch.stack(rows).mean(dim=0).tolist()
                for key, rows in sorted(buckets.items())
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
