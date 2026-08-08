"""Checkpointed episodic training and K-shot evaluation on verified real caches."""

from __future__ import annotations

import json
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
import yaml

from mmer.config import load_label_mapping
from mmer.engine import evaluate_model
from mmer.meta import EpisodeSampler
from mmer.meta.engine import evaluate_meta_episodes, train_meta_epoch
from mmer.models.trimodal import TrimodalEmotionModel, parameter_counts
from mmer.real_runner import (
    _bundle,
    _class_weights,
    _device,
    _limitations,
    _resolve_config,
    _source_snapshot_sha256,
    _versions,
)
from mmer.runner import _git_hash, _inside_root, _loader, _model_kwargs
from mmer.utils import read_checkpoint, save_checkpoint, seed_everything


def _meta(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("meta")
    if not isinstance(settings, dict) or settings.get("enabled") is not True:
        raise ValueError("episodic runner requires meta.enabled: true")
    required = {
        "n_way",
        "k_shot",
        "query_per_class",
        "episodes_per_epoch",
        "validation_episodes",
        "evaluation_episodes",
        "task_field",
        "distance",
        "prototype_temperature",
        "lambda_classification",
        "lambda_episode",
        "lambda_supcon",
        "supcon_temperature",
        "evaluation_k_shots",
        "primary_k_shot",
    }
    missing = required - set(settings)
    if missing:
        raise ValueError(f"meta configuration is missing: {sorted(missing)}")
    positive_counts = (
        "k_shot",
        "query_per_class",
        "episodes_per_epoch",
        "validation_episodes",
        "evaluation_episodes",
    )
    if int(settings["n_way"]) <= 1 or any(int(settings[name]) <= 0 for name in positive_counts):
        raise ValueError("meta episode counts must be positive and n_way must exceed one")
    if str(settings["distance"]) not in {"cosine", "squared_euclidean"}:
        raise ValueError("meta distance must be cosine or squared_euclidean")
    if float(settings["prototype_temperature"]) <= 0 or float(settings["supcon_temperature"]) <= 0:
        raise ValueError("meta temperatures must be positive")
    loss_weights = [
        float(settings[name])
        for name in ("lambda_classification", "lambda_episode", "lambda_supcon")
    ]
    if min(loss_weights) < 0 or sum(loss_weights) <= 0:
        raise ValueError("meta loss weights must be non-negative with a positive total")
    k_values = [int(value) for value in settings["evaluation_k_shots"]]
    if any(value <= 0 for value in k_values) or len(set(k_values)) != len(k_values):
        raise ValueError("evaluation_k_shots must contain unique positive integers")
    if int(settings["primary_k_shot"]) not in k_values:
        raise ValueError("primary_k_shot must be included in evaluation_k_shots")
    return settings


def _sampler(
    dataset: object,
    settings: dict[str, Any],
    seed: int,
    episodes: int,
    k_shot: int | None = None,
    disjoint_speakers: bool = False,
) -> EpisodeSampler:
    return EpisodeSampler(
        dataset=dataset,  # type: ignore[arg-type]
        n_way=int(settings["n_way"]),
        k_shot=int(settings["k_shot"] if k_shot is None else k_shot),
        query_per_class=int(settings["query_per_class"]),
        episodes=int(episodes),
        seed=int(seed),
        task_field=str(settings["task_field"]),
        disjoint_speakers=bool(disjoint_speakers),
    )


def _domain_protocol_audit(settings: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    split_counts = audit.get("split_corpus_counts", {})
    train = set(split_counts.get("train", {}))
    validation = set(split_counts.get("validation", {}))
    test = set(split_counts.get("test", {}))
    overlap = {
        "train_validation": sorted(train & validation),
        "train_test": sorted(train & test),
        "validation_test": sorted(validation & test),
    }
    require_unseen = bool(settings.get("require_unseen_test_corpus", False))
    if require_unseen and (not train or not test or overlap["train_test"]):
        raise ValueError(
            "require_unseen_test_corpus needs non-empty disjoint train/test corpus sets; "
            f"overlap={overlap['train_test']}"
        )
    return {
        "require_unseen_test_corpus": require_unseen,
        "train_corpora": sorted(train),
        "validation_corpora": sorted(validation),
        "test_corpora": sorted(test),
        "corpus_overlap": overlap,
        "unseen_test_corpus_verified": bool(train and test and not overlap["train_test"]),
    }


def run_meta_training(
    config_path: str | Path,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Train projections/adapters/fusion using balanced cached-embedding episodes."""

    root = Path(project_root).resolve()
    config_file, config = _resolve_config(config_path, root)
    settings = _meta(config)
    source_snapshot = _source_snapshot_sha256(root, config_file)
    labels = load_label_mapping(_inside_root(root, config["labels_path"], "label mapping"))
    seed = int(config["seed"])
    seed_everything(seed)
    device = _device(config)
    bundle = _bundle(root, config, labels)
    domain_protocol = _domain_protocol_audit(settings, bundle.audit)
    train_set = bundle.splits["train"]
    validation_set = bundle.splits["validation"]
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
    output_dir = _inside_root(root, config["output_dir"], "output directory")
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
    dropout_generator = torch.Generator().manual_seed(seed + 100)

    validation_sampler = _sampler(
        validation_set,
        settings,
        seed + 20_000,
        int(settings["validation_episodes"]),
        disjoint_speakers=bool(settings.get("disjoint_speakers_validation", True)),
    )
    for epoch in range(1, int(config["epochs"]) + 1):
        train_sampler = _sampler(
            train_set,
            settings,
            seed + epoch * 1_003,
            int(settings["episodes_per_epoch"]),
            disjoint_speakers=bool(settings.get("disjoint_speakers_train", False)),
        )
        losses = train_meta_epoch(
            model,
            train_sampler,
            optimizer,
            device,
            distance=str(settings["distance"]),
            prototype_temperature=float(settings["prototype_temperature"]),
            lambda_classification=float(settings["lambda_classification"]),
            lambda_episode=float(settings["lambda_episode"]),
            lambda_supcon=float(settings["lambda_supcon"]),
            supcon_temperature=float(settings["supcon_temperature"]),
            modality_dropout=float(config.get("modality_dropout", 0.0)),
            class_weights=weights,
            dropout_generator=dropout_generator,
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
            mixed_precision=config.get("mixed_precision", False),
        )
        validation = evaluate_meta_episodes(
            model,
            validation_sampler,
            device,
            len(labels),
            str(settings["distance"]),
            float(settings["prototype_temperature"]),
            mixed_precision=config.get("mixed_precision", False),
        )
        history.append({"epoch": epoch, "losses": losses, "validation": validation})
        current_uar = float(validation["uar"]["mean"])
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
                    "training_mode": "episodic",
                    "pilot": bool(config.get("pilot", False)),
                    "model_kwargs": kwargs,
                    "meta": settings,
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

    payload = read_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    zero_shot_validation = evaluate_model(
        model,
        _loader(validation_set, int(config["batch_size"]), False, seed),
        device,
        len(labels),
        mixed_precision=config.get("mixed_precision", False),
    )
    metadata = {
        "synthetic": False,
        "training_mode": "episodic",
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
        "optimization": {
            "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
            "mixed_precision": config.get("mixed_precision", False),
        },
        "meta_config": settings,
        "domain_protocol": domain_protocol,
        "train_episode_audit": _sampler(
            train_set,
            settings,
            seed + 1_003,
            int(settings["episodes_per_epoch"]),
            disjoint_speakers=bool(settings.get("disjoint_speakers_train", False)),
        ).audit(),
        "validation_episode_audit": validation_sampler.audit(),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_uar": best_uar,
        "best_validation": history[best_epoch - 1]["validation"],
        "zero_shot_validation": zero_shot_validation,
        "training_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "history": history,
        "limitations": _limitations(config, bundle.audit)
        + ["single-corpus episodic engineering run; not unseen-corpus evidence"],
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, allow_unicode=True)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def run_meta_evaluation(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    project_root: str | Path = ".",
    modality_subsets: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Evaluate zero-shot and speaker-disjoint 1/5/10-shot target episodes."""

    root = Path(project_root).resolve()
    config_file, config = _resolve_config(config_path, root)
    settings = _meta(config)
    source_snapshot = _source_snapshot_sha256(root, config_file)
    labels = load_label_mapping(_inside_root(root, config["labels_path"], "label mapping"))
    seed = int(config["seed"])
    seed_everything(seed)
    device = _device(config)
    bundle = _bundle(root, config, labels)
    domain_protocol = _domain_protocol_audit(settings, bundle.audit)
    path_value = checkpoint_path or Path(config["output_dir"]) / "best_checkpoint.pt"
    path = _inside_root(root, path_value, "checkpoint")
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    payload = read_checkpoint(path, map_location=device)
    checkpoint_metadata = payload["metadata"]
    kwargs = _model_kwargs(config, len(labels))
    checks = {
        "training_mode": checkpoint_metadata.get("training_mode") == "episodic",
        "manifest": checkpoint_metadata.get("manifest_sha256") == bundle.audit["manifest_sha256"],
        "caches": checkpoint_metadata.get("cache_contracts") == bundle.audit["cache_contracts"],
        "labels": checkpoint_metadata.get("labels") == labels,
        "model": checkpoint_metadata.get("model_kwargs") == kwargs,
        "meta": checkpoint_metadata.get("meta") == settings,
    }
    failed = [name for name, value in checks.items() if not value]
    if failed:
        raise ValueError(f"episodic checkpoint contract mismatch: {failed}")
    model = TrimodalEmotionModel(**kwargs).to(device)
    model.load_state_dict(payload["model_state"])
    test_set = bundle.splits["test"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    zero_shot = evaluate_model(
        model,
        _loader(test_set, int(config["batch_size"]), False, seed),
        device,
        len(labels),
        mixed_precision=config.get("mixed_precision", False),
    )
    few_shot: dict[str, Any] = {}
    for offset, k_shot in enumerate(int(value) for value in settings["evaluation_k_shots"]):
        sampler = _sampler(
            test_set,
            settings,
            seed + 50_000 + offset * 10_000,
            int(settings["evaluation_episodes"]),
            k_shot=k_shot,
            disjoint_speakers=bool(settings.get("disjoint_speakers_test", True)),
        )
        result = evaluate_meta_episodes(
            model,
            sampler,
            device,
            len(labels),
            str(settings["distance"]),
            float(settings["prototype_temperature"]),
            mixed_precision=config.get("mixed_precision", False),
        )
        result["episode_audit"] = sampler.audit()
        few_shot[str(k_shot)] = result
    primary_k = str(int(settings["primary_k_shot"]))
    primary = few_shot[primary_k]
    modality_stress: dict[str, Any] = {}
    if modality_subsets:
        primary_offset = [int(value) for value in settings["evaluation_k_shots"]].index(
            int(primary_k)
        )
        for subset in modality_subsets:
            key = "+".join(str(value) for value in subset)
            if not key or key in modality_stress:
                raise ValueError("modality stress subsets must be non-empty and unique")
            stress_sampler = _sampler(
                test_set,
                settings,
                seed + 50_000 + primary_offset * 10_000,
                int(settings["evaluation_episodes"]),
                k_shot=int(primary_k),
                disjoint_speakers=bool(settings.get("disjoint_speakers_test", True)),
            )
            modality_stress[key] = {
                "zero_shot": evaluate_model(
                    model,
                    _loader(test_set, int(config["batch_size"]), False, seed),
                    device,
                    len(labels),
                    modality_subset=list(subset),
                    mixed_precision=config.get("mixed_precision", False),
                ),
                f"{primary_k}_shot": evaluate_meta_episodes(
                    model,
                    stress_sampler,
                    device,
                    len(labels),
                    str(settings["distance"]),
                    float(settings["prototype_temperature"]),
                    modality_subset=list(subset),
                    mixed_precision=config.get("mixed_precision", False),
                ),
            }
    metrics = {
        "synthetic": False,
        "training_mode": "episodic",
        "protocol": (
            f"unseen_corpus_speaker_disjoint_{primary_k}_shot_episode_mean"
            if domain_protocol["unseen_test_corpus_verified"]
            else f"speaker_disjoint_{primary_k}_shot_episode_mean"
        ),
        "uar": float(primary["uar"]["mean"]),
        "macro_f1": float(primary["macro_f1"]["mean"]),
        "accuracy": float(primary["accuracy"]["mean"]),
        "zero_shot": zero_shot,
        "few_shot": few_shot,
        "primary_k_shot": int(primary_k),
        "modality_stress": modality_stress,
        "checkpoint_epoch": int(payload["epoch"]),
        "source_snapshot_sha256": source_snapshot,
        "source_snapshot_matches_checkpoint": (
            checkpoint_metadata.get("source_snapshot_sha256") == source_snapshot
        ),
        "inference_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "parameter_counts": parameter_counts(model),
        "data_audit": bundle.audit,
        "domain_protocol": domain_protocol,
        "paper_ready": False,
        "limitations": _limitations(config, bundle.audit)
        + ["single-corpus episodic engineering run; not unseen-corpus evidence"],
    }
    output_dir = _inside_root(root, config["output_dir"], "output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
