#!/usr/bin/env python
"""Train the iteration-1 synthetic cached-embedding model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.runner import run_smoke_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/cpu_smoke.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    summary = run_smoke_training(args.config, args.project_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
