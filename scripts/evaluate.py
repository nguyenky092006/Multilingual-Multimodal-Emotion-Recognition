#!/usr/bin/env python
"""Reload and evaluate either a synthetic or verified real-cache checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.config import load_yaml
from mmer.meta_runner import run_meta_evaluation
from mmer.real_runner import run_cached_evaluation
from mmer.runner import run_smoke_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/cpu_smoke.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_yaml(config_path)
    meta = config.get("meta")
    if isinstance(meta, dict) and bool(meta.get("enabled", False)):
        metrics = run_meta_evaluation(args.config, args.checkpoint, root)
    elif bool(config.get("data", {}).get("synthetic", False)):
        metrics = run_smoke_evaluation(args.config, args.checkpoint, root)
    else:
        metrics = run_cached_evaluation(args.config, args.checkpoint, root)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
