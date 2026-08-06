#!/usr/bin/env python
"""Validate a JSONL manifest and fail on serious leakage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mmer.config import load_label_mapping
from mmer.data import load_manifest, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("configs/data/labels.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--skip-file-checks", action="store_true")
    args = parser.parse_args()
    labels = load_label_mapping(args.labels)
    samples = load_manifest(args.manifest)
    report = validate_manifest(
        samples, set(labels), root=args.data_root, check_files=not args.skip_file_checks
    )
    print(
        json.dumps(
            {
                "sample_count": report.sample_count,
                "is_valid": report.is_valid,
                "language_corpus_counts": report.language_corpus_counts,
                "issues": [asdict(issue) for issue in report.issues],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

