#!/usr/bin/env python
"""Build deterministic leakage-safe full and pilot EmotionTalk manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.config import load_yaml
from mmer.data import build_emotiontalk_manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/emotiontalk.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-file-checks", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    ratios = config.get("split_ratios", [0.70, 0.15, 0.15])
    pilot_counts = config.get(
        "pilot_samples_per_class", {"train": 64, "validation": 16, "test": 16}
    )
    result = build_emotiontalk_manifests(
        project_root=args.project_root,
        dataset_root=args.dataset_root or config.get("dataset_root", "data/raw/emotiontalk"),
        output_dir=args.output_dir or config.get("output_dir", "data/manifests"),
        seed=int(config.get("seed", 17)),
        split_ratios=tuple(float(value) for value in ratios),
        pilot_samples_per_class={str(key): int(value) for key, value in pilot_counts.items()},
        check_files=not args.skip_file_checks,
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
