"""Training and checkpoint-reload evaluation for verified real embedding caches."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mmer.config import load_label_mapping, load_yaml
from mmer.data.real_cached import RealCacheBundle, load_real_cache_bundle
from mmer.engine import evaluate_model, train_one_epoch
from mmer.models.trimodal import TrimodalEmotionModel, parameter_counts
from mmer.runner import _git_hash, _loader, _model_kwargs
from mmer.utils import load_checkpoint, save_checkpoint, seed_everything


def _resolve_config(config_path: str | Path, root: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(config_path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"configuration must be inside project root: {path}") from exc
    config = load_yaml(path)
    if config.get("data", {}).get("synthetic", True):
        raise ValueError("real cached runner requires data.synthetic: false")
    return path, config


def _device(config: dict[str, Any]) -> torch.device:
    device = torch.device(str(config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    return device


def _bundle(root: Path, config: dict[str, Any], labels: dict[str, int]) -> RealCacheBundle:
    data = config["data"]
    return load_real_cache_bundle(
        project_root=root,
        manifest_path=data["manifest_path"],
        cache_sources=data["caches"],
        labels=labels,
        input_dims=data["input_dims"],
        enabled_modalities=data.get("enabled_modalities", ["audio", "text", "visual"]),
        quality_policy=str(data.get("quality_policy", "validated_metadata_v1")),
    )


def _class_weights(dataset: object, num_classes: int) -> torch.Tensor:
    examples = getattr(dataset, "examples", None)
    if not isinstance(examples, list):
        raise ValueError("class weighting requires a materialised cached dataset")
    counts = torch.bincount(
        torch.tensor([int(item.label) for item in examples], dtype=torch.long),
        minlength=num_classes,
    ).float()
    if (counts == 0).any():
        raise ValueError(f"training split has an empty class: {counts.tolist()}")
    return counts.sum() / (num_classes * counts)


def _versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
    }
    for package in ("safetensors", "transformers", "av"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _source_snapshot_sha256(root: Path, config_path: Path) -> str:
    """Hash executable project sources and the exact experiment configuration."""

    paths = [
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        root / "pyproject.toml",
        config_path,
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def run_cached_training(
    config_path: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Train adapters/fusion/classifier on verified frozen real embeddings."""

    root = Path(project_root).resolve()
    config_file, config = _resolve_config(config_path, root)
    source_snapshot = _source_snapshot_sha256(root, config_file)
    labels = load_label_mapping(root / config["labels_path"])
    seed = int(config["seed"])
    seed_everything(seed)
    device = _device(config)
    bundle = _bundle(root, config, labels)
    train_set = bundle.splits["train"]
    validation_set = bundle.splits["validation"]
    batch_size = int(config["batch_size"])
    train_loader = _loader(train_set, batch_size, True, seed)
    validation_loader = _loader(validation_set, batch_size, False, seed)

    kwargs = _model_kwargs(config, len(labels))
    model = TrimodalEmotionModel(**kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    weights = None
    if bool(config.get("use_class_weights", False)):
        weights = _class_weights(train_set, len(labels)).to(device)
    dropout_generator = torch.Generator().manual_seed(seed + 100)

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_uar = -1.0
    best_epoch = 0
    stale_epochs = 0
    patience = int(config.get("early_stopping_patience", int(config["epochs"])))
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive")

    for epoch in range(1, int(config["epochs"]) + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            float(config.get("modality_dropout", 0.0)),
            class_weights=weights,
            dropout_generator=dropout_generator,
        )
        validation_metrics = evaluate_model(model, validation_loader, device, len(labels))
        history.append({"epoch": epoch, "train_loss": loss, "validation": validation_metrics})
        current_uar = float(validation_metrics["uar"])
        if current_uar > best_uar:
            best_uar = current_uar
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                {
                    "synthetic": False,
                    "pilot": bool(config.get("pilot", False)),
                    "model_kwargs": kwargs,
                    "labels": labels,
                    "config": config,
                    "manifest_sha256": bundle.audit["manifest_sha256"],
                    "cache_contracts": bundle.audit["cache_contracts"],
                    "source_snapshot_sha256": source_snapshot,
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed = time.perf_counter() - started
    best_validation = history[best_epoch - 1]["validation"]
    label_names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    zero_recall_classes = [
        label_names[index]
        for index, value in enumerate(best_validation["per_class_recall"])
        if float(value) == 0.0
    ]
    global_weights = [float(value) for value in best_validation["fusion_weight_global_mean"]]
    dominant_index = max(range(len(global_weights)), key=global_weights.__getitem__)
    dominant_modality = ("audio", "text", "visual")[dominant_index]
    multiple_modalities = len(config["data"].get("enabled_modalities", [])) > 1
    diagnostic_flags = {
        "zero_recall_classes_at_best_validation": zero_recall_classes,
        "dominant_fusion_modality": (
            dominant_modality
            if multiple_modalities and global_weights[dominant_index] > 0.7
            else None
        ),
        "dominant_fusion_weight": global_weights[dominant_index],
    }
    metadata = {
        "synthetic": False,
        "pilot": bool(config.get("pilot", False)),
        "diagnostic": bool(config.get("diagnostic", False)),
        "paper_ready": False,
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "git_commit": _git_hash(root),
        "source_snapshot_sha256": source_snapshot,
        "versions": _versions(),
        "label_mapping": labels,
        "data_audit": bundle.audit,
        "parameter_counts": parameter_counts(model),
        "class_weights": None if weights is None else weights.detach().cpu().tolist(),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_uar": best_uar,
        "training_seconds": elapsed,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "history": history,
        "diagnostic_flags": diagnostic_flags,
        "limitations": [
            "single English corpus",
            "pilot actor subset",
            "single seed unless repeated explicitly",
            "CREMA-D text contains only twelve fixed official prompts",
            "not a paper-ready comparison",
        ],
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, allow_unicode=True)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def run_cached_evaluation(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Reload the best real-cache checkpoint and evaluate its held-out test speakers."""

    root = Path(project_root).resolve()
    config_file, config = _resolve_config(config_path, root)
    current_source_snapshot = _source_snapshot_sha256(root, config_file)
    labels = load_label_mapping(root / config["labels_path"])
    seed_everything(int(config["seed"]))
    device = _device(config)
    bundle = _bundle(root, config, labels)
    path = Path(checkpoint_path) if checkpoint_path else root / config["output_dir"] / "best_checkpoint.pt"
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")

    provisional = TrimodalEmotionModel(**_model_kwargs(config, len(labels))).to(device)
    payload = load_checkpoint(path, provisional, map_location=device)
    checkpoint_metadata = payload["metadata"]
    if checkpoint_metadata.get("synthetic", True):
        raise ValueError("refusing to evaluate a synthetic checkpoint as real")
    if checkpoint_metadata.get("manifest_sha256") != bundle.audit["manifest_sha256"]:
        raise ValueError("checkpoint manifest hash does not match current real cache bundle")
    if checkpoint_metadata.get("labels") != labels:
        raise ValueError("checkpoint label mapping differs from current configuration")

    loader = _loader(bundle.splits["test"], int(config["batch_size"]), False, int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = evaluate_model(provisional, loader, device, len(labels))
    metrics.update(
        {
            "synthetic": False,
            "pilot": bool(config.get("pilot", False)),
            "diagnostic": bool(config.get("diagnostic", False)),
            "paper_ready": False,
            "source_snapshot_sha256": current_source_snapshot,
            "source_snapshot_matches_checkpoint": (
                checkpoint_metadata.get("source_snapshot_sha256") == current_source_snapshot
            ),
            "checkpoint_epoch": int(payload["epoch"]),
            "inference_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "parameter_counts": parameter_counts(provisional),
            "data_audit": bundle.audit,
        }
    )
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
