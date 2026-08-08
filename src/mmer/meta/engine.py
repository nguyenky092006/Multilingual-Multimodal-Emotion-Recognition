"""Training and evaluation loops for episodic cached-embedding learning."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from mmer.data.cached import apply_modality_dropout, apply_modality_subset, collate_cached
from mmer.engine import _step_optimizer, _to_device, mixed_precision_dtype
from mmer.metrics import classification_metrics
from mmer.models.trimodal import TrimodalEmotionModel

from .episodes import Episode
from .losses import supervised_contrastive_loss
from .prototypical import prototypical_loss


def train_meta_epoch(
    model: TrimodalEmotionModel,
    episodes: Iterable[Episode],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    distance: str,
    prototype_temperature: float,
    lambda_classification: float,
    lambda_episode: float,
    lambda_supcon: float,
    supcon_temperature: float,
    modality_dropout: float = 0.0,
    class_weights: Tensor | None = None,
    dropout_generator: torch.Generator | None = None,
    gradient_accumulation_steps: int = 1,
    mixed_precision: object = False,
) -> dict[str, float]:
    """Optimize classification, prototype, and optional shared-space contrastive losses."""

    if min(lambda_classification, lambda_episode, lambda_supcon) < 0:
        raise ValueError("meta loss weights must be non-negative")
    if lambda_classification + lambda_episode + lambda_supcon <= 0:
        raise ValueError("at least one meta loss weight must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train()
    dtype = mixed_precision_dtype(mixed_precision, device)
    scaler = torch.amp.GradScaler(
        device.type, enabled=dtype == torch.float16 and device.type == "cuda"
    )
    classification_loss = nn.CrossEntropyLoss(weight=class_weights)
    totals = {"total": 0.0, "classification": 0.0, "episode": 0.0, "supcon": 0.0}
    count = 0
    accumulated = 0
    optimizer.zero_grad(set_to_none=True)
    for episode in episodes:
        support = _to_device(collate_cached(episode.support), device)
        query = _to_device(collate_cached(episode.query), device)
        support_mask = apply_modality_dropout(
            support["modality_mask"], modality_dropout, dropout_generator
        )
        query_mask = apply_modality_dropout(
            query["modality_mask"], modality_dropout, dropout_generator
        )
        with torch.autocast(
            device_type=device.type,
            dtype=dtype or torch.float32,
            enabled=dtype is not None,
        ):
            support_output = model(
                support["embeddings"],
                support_mask,
                support["languages"],
                support["corpora"],
                support["quality"],
                support["temporal_masks"],
            )
            query_output = model(
                query["embeddings"],
                query_mask,
                query["languages"],
                query["corpora"],
                query["quality"],
                query["temporal_masks"],
            )
            classes = torch.tensor(episode.classes, dtype=torch.long, device=device)
            proto = prototypical_loss(
                support_output["fused"],
                support["labels"],
                query_output["fused"],
                query["labels"],
                classes=classes,
                distance=distance,
                temperature=prototype_temperature,
            )
            cls = classification_loss(query_output["logits"], query["labels"])
            contrastive = supervised_contrastive_loss(
                torch.cat([support_output["shared_fused"], query_output["shared_fused"]]),
                torch.cat([support["labels"], query["labels"]]),
                temperature=supcon_temperature,
            )
            loss = (
                lambda_classification * cls
                + lambda_episode * proto.loss
                + lambda_supcon * contrastive
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite episodic training loss: {loss.item()}")
        scaler.scale(loss / gradient_accumulation_steps).backward()
        accumulated += 1
        if accumulated == gradient_accumulation_steps:
            _step_optimizer(optimizer, scaler, accumulated, gradient_accumulation_steps)
            accumulated = 0
        values = {
            "total": loss,
            "classification": cls,
            "episode": proto.loss,
            "supcon": contrastive,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu())
        count += 1
    if count == 0:
        raise ValueError("episodic training produced no episodes")
    if accumulated:
        _step_optimizer(optimizer, scaler, accumulated, gradient_accumulation_steps)
    return {name: value / count for name, value in totals.items()}


def _aggregate(records: list[dict[str, Any]], name: str) -> dict[str, float]:
    values = np.asarray([float(record[name]) for record in records], dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": max(0.0, mean - half_width),
        "ci95_high": min(1.0, mean + half_width),
    }


@torch.no_grad()
def evaluate_meta_episodes(
    model: TrimodalEmotionModel,
    episodes: Iterable[Episode],
    device: torch.device,
    num_classes: int,
    distance: str,
    prototype_temperature: float,
    modality_subset: list[str] | tuple[str, ...] | None = None,
    mixed_precision: object = False,
) -> dict[str, Any]:
    """Evaluate prototypes and retain support/query provenance for leakage auditing."""

    model.eval()
    dtype = mixed_precision_dtype(mixed_precision, device)
    records: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        support = _to_device(collate_cached(episode.support), device)
        query = _to_device(collate_cached(episode.query), device)
        support_mask = apply_modality_subset(support["modality_mask"], modality_subset)
        query_mask = apply_modality_subset(query["modality_mask"], modality_subset)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype or torch.float32,
            enabled=dtype is not None,
        ):
            support_output = model(
                support["embeddings"],
                support_mask,
                support["languages"],
                support["corpora"],
                support["quality"],
                support["temporal_masks"],
            )
            query_output = model(
                query["embeddings"],
                query_mask,
                query["languages"],
                query["corpora"],
                query["quality"],
                query["temporal_masks"],
            )
            result = prototypical_loss(
                support_output["fused"],
                support["labels"],
                query_output["fused"],
                query["labels"],
                classes=torch.tensor(episode.classes, dtype=torch.long, device=device),
                distance=distance,
                temperature=prototype_temperature,
            )
        metrics = classification_metrics(
            query["labels"].cpu().tolist(), result.predictions.cpu().tolist(), num_classes
        )
        records.append(
            {
                "episode": index,
                "task_key": episode.task_key,
                "support_speakers": list(episode.support_speakers),
                "query_speakers": list(episode.query_speakers),
                "support_query_sample_overlap": bool(
                    {item.sample_id for item in episode.support}
                    & {item.sample_id for item in episode.query}
                ),
                "support_query_speaker_overlap": bool(
                    set(episode.support_speakers) & set(episode.query_speakers)
                ),
                "prototype_loss": float(result.loss.cpu()),
                **metrics,
            }
        )
    if not records:
        raise ValueError("episodic evaluation produced no episodes")
    if any(record["support_query_sample_overlap"] for record in records):
        raise RuntimeError("support/query utterance leakage detected during evaluation")
    return {
        "episodes": len(records),
        "uar": _aggregate(records, "uar"),
        "macro_f1": _aggregate(records, "macro_f1"),
        "accuracy": _aggregate(records, "accuracy"),
        "prototype_loss": _aggregate(records, "prototype_loss"),
        "support_query_sample_overlap_detected": False,
        "support_query_speaker_overlap_episodes": sum(
            bool(record["support_query_speaker_overlap"]) for record in records
        ),
        "forced_modality_subset": (
            list(modality_subset) if modality_subset is not None else None
        ),
        "episode_records": records,
    }
