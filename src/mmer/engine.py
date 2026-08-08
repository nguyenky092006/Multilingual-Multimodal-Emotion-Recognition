"""Offline cached-embedding train and evaluation loops."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mmer.data.cached import apply_modality_dropout, apply_modality_subset
from mmer.metrics import classification_metrics
from mmer.models.trimodal import TrimodalEmotionModel


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    copied = dict(batch)
    copied["embeddings"] = {
        key: value.to(device) for key, value in batch["embeddings"].items()
    }
    copied["temporal_masks"] = {
        key: value.to(device) for key, value in batch.get("temporal_masks", {}).items()
    }
    for key in ("modality_mask", "quality", "labels"):
        copied[key] = batch[key].to(device)
    return copied


def mixed_precision_dtype(setting: object, device: torch.device) -> torch.dtype | None:
    """Resolve an explicit cached-head precision policy without silent fallback."""

    if setting in (None, False, "none", "false"):
        return None
    if setting is True:
        if device.type != "cuda":
            raise ValueError("mixed_precision=true requires CUDA; use bfloat16 explicitly on CPU")
        return torch.float16
    if setting == "float16":
        if device.type != "cuda":
            raise ValueError("float16 training requires CUDA")
        return torch.float16
    if setting == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16")
        return torch.bfloat16
    raise ValueError("mixed_precision must be false, true, float16, or bfloat16")


def _step_optimizer(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    accumulated: int,
    target_accumulation: int,
) -> None:
    scaler.unscale_(optimizer)
    if accumulated != target_accumulation:
        correction = target_accumulation / accumulated
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_one_epoch(
    model: TrimodalEmotionModel,
    loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality_dropout: float = 0.0,
    class_weights: Tensor | None = None,
    dropout_generator: torch.Generator | None = None,
    gradient_accumulation_steps: int = 1,
    mixed_precision: object = False,
) -> float:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train()
    dtype = mixed_precision_dtype(mixed_precision, device)
    scaler = torch.amp.GradScaler(
        device.type, enabled=dtype == torch.float16 and device.type == "cuda"
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    batch_count = 0
    accumulated = 0
    optimizer.zero_grad(set_to_none=True)
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        mask = apply_modality_dropout(
            batch["modality_mask"], modality_dropout, dropout_generator
        )
        with torch.autocast(
            device_type=device.type,
            dtype=dtype or torch.float32,
            enabled=dtype is not None,
        ):
            output = model(
                batch["embeddings"],
                mask,
                batch["languages"],
                batch["corpora"],
                batch["quality"],
                batch["temporal_masks"],
            )
            loss = loss_function(output["logits"], batch["labels"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        scaler.scale(loss / gradient_accumulation_steps).backward()
        accumulated += 1
        if accumulated == gradient_accumulation_steps:
            _step_optimizer(optimizer, scaler, accumulated, gradient_accumulation_steps)
            accumulated = 0
        total_loss += float(loss.detach().cpu())
        batch_count += 1
    if batch_count == 0:
        raise ValueError("training loader produced no batches")
    if accumulated:
        _step_optimizer(optimizer, scaler, accumulated, gradient_accumulation_steps)
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


def _summarise_route_stats(records: Sequence[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for record in records:
        for modality, raw_components in record.items():
            if not isinstance(raw_components, dict):
                continue
            modality_summary = summary.setdefault(modality, {})
            assert isinstance(modality_summary, dict)
            for component, raw_stats in raw_components.items():
                if not isinstance(raw_stats, dict):
                    continue
                component_summary = modality_summary.setdefault(
                    component,
                    {
                        "enabled": False,
                        "language_usage": Counter(),
                        "corpus_usage": Counter(),
                        "language_output_norm": defaultdict(list),
                        "corpus_output_norm": defaultdict(list),
                        "output_norm": [],
                        "collapse_detected": False,
                    },
                )
                component_summary["enabled"] = bool(
                    component_summary["enabled"] or raw_stats.get("enabled", False)
                )
                component_summary["collapse_detected"] = bool(
                    component_summary["collapse_detected"]
                    or raw_stats.get("collapse_detected", False)
                )
                for key in ("language_usage", "corpus_usage"):
                    values = raw_stats.get(key, {})
                    if isinstance(values, dict):
                        component_summary[key].update(
                            {str(name): int(value) for name, value in values.items()}
                        )
                for key in ("language_output_norm", "corpus_output_norm"):
                    values = raw_stats.get(key, {})
                    if isinstance(values, dict):
                        for name, value in values.items():
                            component_summary[key][str(name)].append(float(value))
                if "output_norm" in raw_stats:
                    component_summary["output_norm"].append(float(raw_stats["output_norm"]))
    for raw_modality in summary.values():
        assert isinstance(raw_modality, dict)
        for raw_component in raw_modality.values():
            for key in ("language_usage", "corpus_usage"):
                raw_component[key] = dict(sorted(raw_component[key].items()))
            for key in ("language_output_norm", "corpus_output_norm"):
                raw_component[key] = {
                    name: sum(values) / len(values)
                    for name, values in sorted(raw_component[key].items())
                }
            values = raw_component["output_norm"]
            raw_component["output_norm"] = sum(values) / len(values) if values else None
    return summary


@torch.no_grad()
def evaluate_model(
    model: TrimodalEmotionModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    num_classes: int,
    modality_subset: Sequence[str] | None = None,
    mixed_precision: object = False,
) -> dict[str, Any]:
    model.eval()
    dtype = mixed_precision_dtype(mixed_precision, device)
    targets: list[int] = []
    predictions: list[int] = []
    all_weights: list[Tensor] = []
    languages: list[str] = []
    corpora: list[str] = []
    emotions: list[str] = []
    masks: list[Tensor] = []
    route_records: list[dict[str, object]] = []
    metadata_groups: dict[str, list[str]] = {
        "transcript_source": [],
        "country": [],
        "region": [],
        "accent": [],
    }
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        mask = apply_modality_subset(batch["modality_mask"], modality_subset)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype or torch.float32,
            enabled=dtype is not None,
        ):
            output = model(
                batch["embeddings"],
                mask,
                batch["languages"],
                batch["corpora"],
                batch["quality"],
                batch["temporal_masks"],
            )
        logits = output["logits"]
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(batch["labels"].cpu().tolist())
        all_weights.append(output["fusion_weights"].cpu())
        masks.append(output["effective_modality_mask"].cpu())
        languages.extend(batch["languages"])
        corpora.extend(batch["corpora"])
        emotions.extend(batch["emotions"])
        route_records.append(output["route_stats"])
        for metadata in raw_batch.get("metadata", [{} for _ in batch["languages"]]):
            for name in metadata_groups:
                metadata_groups[name].append(str(metadata.get(name) or "unknown"))
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
        **{
            name: _grouped_metrics(targets, predictions, values, num_classes)
            for name, values in metadata_groups.items()
            if len(values) == len(targets)
        },
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
    metrics["adapter_route_summary"] = _summarise_route_stats(route_records)
    metrics["forced_modality_subset"] = (
        list(modality_subset) if modality_subset is not None else None
    )
    return metrics
