#!/usr/bin/env python
"""Compare trainable parameter budgets without loading datasets or encoder weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.config import load_label_mapping, load_yaml
from mmer.models.trimodal import TrimodalEmotionModel, parameter_counts
from mmer.runner import _model_kwargs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--reference", type=int, default=0)
    parser.add_argument("--max-relative-gap", type=float)
    args = parser.parse_args()
    root = args.project_root.resolve()
    records = []
    for value in args.configs:
        path = value if value.is_absolute() else root / value
        config = load_yaml(path)
        labels_path = Path(config["labels_path"])
        labels_path = labels_path if labels_path.is_absolute() else root / labels_path
        labels = load_label_mapping(labels_path)
        model = TrimodalEmotionModel(**_model_kwargs(config, len(labels)))
        records.append(
            {
                "config": path.relative_to(root).as_posix(),
                "experiment_name": config["experiment_name"],
                **parameter_counts(model),
            }
        )
    if not 0 <= args.reference < len(records):
        raise ValueError("--reference index is out of range")
    reference = int(records[args.reference]["trainable"])
    for record in records:
        difference = int(record["trainable"]) - reference
        record["difference_from_reference"] = difference
        record["relative_gap"] = abs(difference) / reference if reference else 0.0
    if args.max_relative_gap is not None and any(
        float(record["relative_gap"]) > args.max_relative_gap for record in records
    ):
        raise RuntimeError("at least one model exceeds the requested parameter-budget gap")
    print(json.dumps({"reference": args.reference, "models": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
