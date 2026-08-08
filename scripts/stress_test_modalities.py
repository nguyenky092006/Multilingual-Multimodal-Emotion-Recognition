#!/usr/bin/env python
"""Evaluate a verified checkpoint under controlled modality removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.config import load_yaml
from mmer.meta_runner import run_meta_evaluation
from mmer.real_runner import run_cached_evaluation


def _default_scenarios(enabled: list[str]) -> list[list[str]]:
    candidates = [[name] for name in enabled]
    if len(enabled) > 2:
        candidates.extend(
            [[name for name in enabled if name != removed] for removed in enabled]
        )
    unique: list[list[str]] = []
    for values in candidates:
        if values not in unique and values != enabled:
            unique.append(values)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--scenario",
        action="append",
        help="Comma-separated available modalities; repeat for multiple scenarios.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_yaml(config_path)
    enabled = [str(value) for value in config["data"]["enabled_modalities"]]
    scenarios = (
        [[value.strip() for value in raw.split(",") if value.strip()] for raw in args.scenario]
        if args.scenario
        else _default_scenarios(enabled)
    )
    if not scenarios:
        raise ValueError("no non-full modality stress scenario is available")
    meta = config.get("meta")
    if isinstance(meta, dict) and bool(meta.get("enabled", False)):
        result = run_meta_evaluation(
            args.config, args.checkpoint, root, modality_subsets=scenarios
        )
    else:
        result = run_cached_evaluation(
            args.config, args.checkpoint, root, modality_subsets=scenarios
        )
    print(json.dumps(result["modality_stress"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
