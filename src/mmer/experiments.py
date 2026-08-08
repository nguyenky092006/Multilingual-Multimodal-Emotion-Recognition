"""Reproducible experiment-matrix helpers."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PAPER_METRICS = ("uar", "macro_f1", "accuracy")


def seed_config(base_config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Clone a config and give the run a seed-specific name and output directory."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    config = copy.deepcopy(base_config)
    name = re.sub(r"_seed\d+$", "", str(config.get("experiment_name", "experiment")))
    config["experiment_name"] = f"{name}_seed{seed}"
    config["seed"] = int(seed)
    output = Path(str(config.get("output_dir", "outputs/experiment")))
    stem = re.sub(r"_seed\d+$", "", output.name)
    config["output_dir"] = (output.parent / f"{stem}_seed{seed}").as_posix()
    return config


def aggregate_seed_metrics(
    records: Sequence[dict[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    """Aggregate compatible seed records with sample std and normal CI95."""

    if not records or len(records) != len(seeds):
        raise ValueError("records and seeds must be non-empty and have equal length")
    protocols = {str(record.get("protocol", "unspecified")) for record in records}
    modes = {str(record.get("training_mode", "supervised")) for record in records}
    if len(protocols) != 1 or len(modes) != 1:
        raise ValueError("seed aggregation cannot mix protocols or training modes")
    metrics: dict[str, dict[str, float]] = {}
    for name in PAPER_METRICS:
        try:
            values = np.asarray([float(record[name]) for record in records], dtype=float)
        except KeyError as exc:
            raise ValueError(f"evaluation record has no {name}") from exc
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        metrics[name] = {
            "mean": mean,
            "std": std,
            "ci95_low": max(0.0, mean - half_width),
            "ci95_high": min(1.0, mean + half_width),
        }
    return {
        "runs": len(records),
        "seeds": [int(seed) for seed in seeds],
        "protocol": next(iter(protocols)),
        "training_mode": next(iter(modes)),
        "paper_ready": len(records) >= 3 and all(record.get("paper_ready") is True for record in records),
        "metrics": metrics,
    }
