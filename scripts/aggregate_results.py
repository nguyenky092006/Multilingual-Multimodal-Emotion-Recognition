#!/usr/bin/env python
"""Aggregate compatible completed metrics without inventing missing runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("uar", "macro_f1", "accuracy")


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": max(0.0, mean - half_width),
        "ci95_high": min(1.0, mean + half_width),
    }


def aggregate(records: list[dict[str, Any]], sources: list[str]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one metrics record is required")
    protocols = {str(item.get("protocol", "unspecified")) for item in records}
    modes = {str(item.get("training_mode", "supervised")) for item in records}
    if len(protocols) != 1 or len(modes) != 1:
        raise ValueError(
            f"refusing to mix protocols={sorted(protocols)} or training modes={sorted(modes)}"
        )
    for source, record in zip(sources, records, strict=True):
        for name in METRICS:
            if name not in record:
                raise ValueError(f"{source} has no {name}")

    result: dict[str, Any] = {
        "runs": len(records),
        "sources": sources,
        "protocol": next(iter(protocols)),
        "training_mode": next(iter(modes)),
        "synthetic": all(item.get("synthetic") is True for item in records),
        "paper_ready": all(item.get("paper_ready") is True for item in records),
        "metrics": {
            name: _summary([float(item[name]) for item in records]) for name in METRICS
        },
    }

    few_shot_records = [item.get("few_shot") for item in records]
    if all(isinstance(item, dict) for item in few_shot_records):
        k_sets = [set(item) for item in few_shot_records]
        if any(values != k_sets[0] for values in k_sets[1:]):
            raise ValueError("few-shot records do not contain the same K values")
        few_shot: dict[str, Any] = {}
        for k_shot in sorted(k_sets[0], key=int):
            few_shot[k_shot] = {
                name: _summary(
                    [float(item[k_shot][name]["mean"]) for item in few_shot_records]
                )
                for name in METRICS
            }
        result["few_shot_across_runs"] = few_shot
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for path in args.paths:
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    result = aggregate(records, [path.as_posix() for path in args.paths])
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
