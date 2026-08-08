#!/usr/bin/env python
"""Train either the offline synthetic smoke model or a verified real-cache experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.config import load_yaml
from mmer.real_runner import run_cached_training
from mmer.runner import run_smoke_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/cpu_smoke.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_yaml(config_path)
    if bool(config.get("data", {}).get("synthetic", False)):
        summary = run_smoke_training(args.config, root)
    else:
        summary = run_cached_training(args.config, root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
