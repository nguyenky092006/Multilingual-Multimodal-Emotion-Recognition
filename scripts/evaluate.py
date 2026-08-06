#!/usr/bin/env python
"""Reload a checkpoint and evaluate the synthetic held-out cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.runner import run_smoke_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/cpu_smoke.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    metrics = run_smoke_evaluation(args.config, args.checkpoint, args.project_root)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

