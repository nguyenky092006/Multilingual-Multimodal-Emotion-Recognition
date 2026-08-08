"""Offline cached-embedding train and evaluation loops."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mmer.data.cached import apply_modality_dropout
from mmer.metrics import classification_metrics
from mmer.models.trimodal import TrimodalEmotionModel


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    copied = dict(batch)
    copied["embeddings"] = {
        key: value.to(device) for key, value in batch["embeddings"].items()
    }
    for key in ("modality_mask", "quality", "labels"):
        copied[key] = batch[key].to(device)
    return copied


def train_one_epoch(
    model: TrimodalEmotionModel,
    loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality_dropout: float = 0.0,
    class_weights: Tensor | None = None,
    dropout_generator: torch.Generator | None = None,
) -> float:
    model.train()
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    batch_count = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        mask = apply_modality_dropout(
            batch["modality_mask"], modality_dropout, dropout_generator
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["embeddings"], mask, batch["languages"], batch["corpora"], batch["quality"]
        )
        loss = loss_function(output["logits"], batch["labels"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batch_count += 1
    if batch_count == 0:
        raise ValueError("training loader produced no batches")
    return total_loss / batch_count


def _grouped_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    groups: Sequence[str],
    num_classes: int,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[str(group)].append(index)
    return {
        name: classification_metrics(
            [targets[index] for index in indices],
            [predictions[index] for index in indices],
            num_classes,
        )
        for name, indices in sorted(grouped.items())
    }


@torch.no_grad()
def evaluate_model(
    model: TrimodalEmotionModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    all_weights: list[Tensor] = []
    languages: list[str] = []
    corpora: list[str] = []
    emotions: list[str] = []
    masks: list[Tensor] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        output = model(
            batch["embeddings"],
            batch["modality_mask"],
            batch["languages"],
            batch["corpora"],
            batch["quality"],
        )
        logits = output["logits"]
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(batch["labels"].cpu().tolist())
        all_weights.append(output["fusion_weights"].cpu())
        masks.append(output["effective_modality_mask"].cpu())
        languages.extend(batch["languages"])
        corpora.extend(batch["corpora"])
        emotions.extend(batch["emotions"])
    if not targets:
        raise ValueError("evaluation loader produced no batches")
    metrics = classification_metrics(targets, predictions, num_classes)
    weights = torch.cat(all_weights)
    modality_masks = torch.cat(masks)
    patterns = [
        "".join("ATV"[index] for index, flag in enumerate(row) if flag)
        for row in modality_masks.tolist()
    ]
    metrics["group_metrics"] = {
        "language": _grouped_metrics(targets, predictions, languages, num_classes),
        "corpus": _grouped_metrics(targets, predictions, corpora, num_classes),
        "modality_pattern": _grouped_metrics(targets, predictions, patterns, num_classes),
    }
    metrics["fusion_weight_summary"] = model.summarise_fusion_weights(
        weights, languages, corpora, emotions, modality_masks
    )
    metrics["fusion_weight_global_mean"] = weights.mean(dim=0).tolist()
    metrics["unavailable_weight_max"] = (
        float(weights.masked_select(~modality_masks).abs().max())
        if (~modality_masks).any()
        else 0.0
    )
    return metrics
