#!/usr/bin/env python
"""Build deterministic full and pilot MMER manifests for CREMA-D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.config import load_yaml
from mmer.data import build_cremad_manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/cremad.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--skip-file-checks", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    ratios = config.get("split_ratios", [0.70, 0.15, 0.15])
    pilot_counts = config.get("pilot_speaker_counts", {"train": 8, "validation": 2, "test": 2})
    result = build_cremad_manifests(
        project_root=args.project_root,
        dataset_root=config.get("dataset_root", "data/raw/crema_d"),
        output_dir=config.get("output_dir", "data/manifests"),
        seed=int(config.get("seed", 17)),
        split_ratios=tuple(float(value) for value in ratios),
        pilot_speaker_counts={str(key): int(value) for key, value in pilot_counts.items()},
        check_files=not args.skip_file_checks,
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
