"""End-to-end synthetic smoke training and checkpoint-reload evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mmer.config import load_label_mapping, load_yaml
from mmer.data.cached import MODALITIES, collate_cached, save_cache
from mmer.data.synthetic import make_synthetic_dataset
from mmer.engine import evaluate_model, train_one_epoch
from mmer.models.trimodal import TrimodalEmotionModel, parameter_counts
from mmer.utils import load_checkpoint, save_checkpoint, seed_everything


def _git_hash(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_dir = root / ".git"
        try:
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                loose_ref = git_dir / head.removeprefix("ref: ")
                if loose_ref.exists():
                    return loose_ref.read_text(encoding="utf-8").strip()
            elif len(head) == 40:
                return head
        except OSError:
            pass
        return "unavailable"


def _inside_root(root: Path, value: str | Path, description: str) -> Path:
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} must be inside project root: {path}") from exc
    return path


def _manifest_hash(config: dict[str, Any]) -> str:
    synthetic_contract = json.dumps(config["data"], sort_keys=True).encode("utf-8")
    return "synthetic:" + hashlib.sha256(synthetic_contract).hexdigest()


def _model_kwargs(config: dict[str, Any], label_count: int) -> dict[str, Any]:
    model = config["model"]
    data = config["data"]
    return {
        "input_dims": data["input_dims"],
        "num_classes": label_count,
        "languages": model.get("languages", ["en", "zh"]),
        "corpora": model.get("corpora", ["synthetic_a", "synthetic_b"]),
        "d_model": int(model["d_model"]),
        "projection_hidden": int(model["projection_hidden"]),
        "adapter_bottleneck": int(model["adapter_bottleneck"]),
        "dropout": float(model["dropout"]),
        "fusion": str(model["fusion"]),
        "shared_adapter": bool(model.get("shared_adapter", True)),
        "routing_alpha": float(model.get("routing_alpha", 0.5)),
        "enabled_modalities": data.get("enabled_modalities", list(MODALITIES)),
        "use_modality_adapters": bool(model.get("use_modality_adapters", True)),
        "emotion_adapter_mode": model.get("emotion_adapter_mode"),
        "use_routed_adapters": bool(model.get("use_routed_adapters", True)),
    }


def _validate_training_config(config: dict[str, Any]) -> None:
    for key in (
        "experiment_name", "seed", "device", "epochs", "batch_size", "learning_rate",
        "weight_decay", "output_dir", "labels_path", "data", "model",
    ):
        if key not in config:
            raise ValueError(f"experiment configuration is missing {key}")
    if int(config["epochs"]) <= 0 or int(config["batch_size"]) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    dropout = float(config.get("modality_dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("modality_dropout must be in [0, 1)")
    if int(config.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("gradient_accumulation_steps is not implemented; use 1")
    if bool(config.get("mixed_precision", False)):
        raise ValueError("mixed_precision training is not implemented for cached heads")


def _loader(dataset: object, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_cached,
        num_workers=0,
        generator=generator,
    )


def run_smoke_training(config_path: str | Path, project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = _inside_root(root, config_path, "configuration")
    config = load_yaml(config_file)
    _validate_training_config(config)
    if not config.get("data", {}).get("synthetic", False):
        raise ValueError("smoke runner only accepts synthetic cached embeddings")
    labels = load_label_mapping(_inside_root(root, config["labels_path"], "label mapping"))
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    output_dir = _inside_root(root, config["output_dir"], "output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = config["data"]["input_dims"]
    train_set = make_synthetic_dataset(int(config["data"]["train_samples"]), dimensions, seed, labels)
    validation_set = make_synthetic_dataset(
        int(config["data"]["validation_samples"]), dimensions, seed + 1, labels
    )
    test_set = make_synthetic_dataset(int(config["data"]["test_samples"]), dimensions, seed + 2, labels)
    for name, dataset in (("train", train_set), ("validation", validation_set), ("test", test_set)):
        save_cache(dataset.examples, output_dir / f"synthetic_{name}_cache.pt")
    batch_size = int(config["batch_size"])
    train_loader = _loader(train_set, batch_size, True, seed)
    validation_loader = _loader(validation_set, batch_size, False, seed)
    kwargs = _model_kwargs(config, len(labels))
    model = TrimodalEmotionModel(**kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    dropout_generator = torch.Generator().manual_seed(seed + 100)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_uar = -1.0
    checkpoint_path = output_dir / "best_checkpoint.pt"
    for epoch in range(1, int(config["epochs"]) + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, float(config.get("modality_dropout", 0.0)),
            dropout_generator=dropout_generator,
        )
        validation_metrics = evaluate_model(model, validation_loader, device, len(labels))
        history.append({"epoch": epoch, "train_loss": loss, "validation": validation_metrics})
        if validation_metrics["uar"] > best_uar:
            best_uar = float(validation_metrics["uar"])
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch,
                {"synthetic": True, "model_kwargs": kwargs, "labels": labels, "config": config},
            )
    metadata = {
        "synthetic": True,
        "seed": seed,
        "git_commit": _git_hash(root),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pyyaml": yaml.__version__,
        },
        "encoder_identifiers": load_yaml(root / "configs/encoder/frozen_encoders.yaml"),
        "label_mapping": labels,
        "manifest_hash": _manifest_hash(config),
        "parameter_counts": parameter_counts(model),
        "training_seconds": time.perf_counter() - started,
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "history": history,
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, allow_unicode=True)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def run_smoke_evaluation(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_yaml(_inside_root(root, config_path, "configuration"))
    _validate_training_config(config)
    if not config.get("data", {}).get("synthetic", False):
        raise ValueError("smoke evaluator only accepts synthetic configurations")
    labels = load_label_mapping(_inside_root(root, config["labels_path"], "label mapping"))
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    path_value = checkpoint_path or Path(config["output_dir"]) / "best_checkpoint.pt"
    path = _inside_root(root, path_value, "checkpoint")
    kwargs = _model_kwargs(config, len(labels))
    model = TrimodalEmotionModel(**kwargs).to(device)
    payload = load_checkpoint(path, model, map_location=device)
    metadata = payload["metadata"]
    if metadata.get("synthetic") is not True:
        raise ValueError("refusing to evaluate a non-synthetic checkpoint as synthetic")
    if metadata.get("model_kwargs") != kwargs or metadata.get("labels") != labels:
        raise ValueError("checkpoint model or label contract differs from configuration")
    test_set = make_synthetic_dataset(
        int(config["data"]["test_samples"]), config["data"]["input_dims"], seed + 2, labels
    )
    test_loader = _loader(test_set, int(config["batch_size"]), False, seed)
    started = time.perf_counter()
    metrics = evaluate_model(model, test_loader, device, len(labels))
    metrics.update(
        {
            "synthetic": True,
            "checkpoint_epoch": int(payload["epoch"]),
            "inference_seconds": time.perf_counter() - started,
            "parameter_counts": parameter_counts(model),
        }
    )
    output_dir = _inside_root(root, config["output_dir"], "output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
