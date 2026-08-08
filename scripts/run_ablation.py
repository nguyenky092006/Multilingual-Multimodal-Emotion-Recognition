#!/usr/bin/env python
"""Materialize, train, and optionally evaluate one registered ablation."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from mmer.config import load_yaml
from mmer.meta_runner import run_meta_evaluation, run_meta_training
from mmer.real_runner import run_cached_evaluation, run_cached_training


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/ablation/ablations.yaml")
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    matrix_path = args.matrix if args.matrix.is_absolute() else root / args.matrix
    matrix = load_yaml(matrix_path)
    entries = matrix.get("ablations")
    if not isinstance(entries, dict) or args.name not in entries:
        raise ValueError(f"unknown ablation: {args.name}")
    entry = entries[args.name]
    if not isinstance(entry, dict):
        raise ValueError(f"invalid ablation record: {args.name}")
    status = str(entry.get("status", "ready"))
    if status.startswith("blocked") or status.startswith("use_"):
        reason = entry.get("reason", status)
        raise RuntimeError(f"ablation {args.name} is not directly runnable: {reason}")
    base_value = entry.get("base_config", matrix.get("default_base_config"))
    if not isinstance(base_value, str):
        raise ValueError(f"ablation {args.name} has no base_config")
    base_path = root / base_value
    config = deep_merge(load_yaml(base_path), entry.get("overrides", {}))
    config["experiment_name"] = f"{config['experiment_name']}_ablation_{args.name}"
    config["track"] = "registered_ablation"
    config["diagnostic"] = True
    config["output_dir"] = f"outputs/ablations/{args.name}_seed{int(config['seed'])}"
    materialized = root / "outputs" / "resolved_ablation_configs" / f"{args.name}.yaml"
    materialized.parent.mkdir(parents=True, exist_ok=True)
    materialized.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    relative = materialized.relative_to(root)
    meta = config.get("meta")
    if isinstance(meta, dict) and bool(meta.get("enabled", False)):
        summary = run_meta_training(relative, root)
        metrics = run_meta_evaluation(relative, project_root=root) if args.evaluate else None
    else:
        summary = run_cached_training(relative, root)
        metrics = run_cached_evaluation(relative, project_root=root) if args.evaluate else None
    print(
        json.dumps(
            {
                "ablation": args.name,
                "registry_status": status,
                "resolved_config": relative.as_posix(),
                "training": summary,
                "evaluation": metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
