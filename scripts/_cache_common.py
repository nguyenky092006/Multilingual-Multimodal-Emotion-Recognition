"""Shared guarded CLI for deferred encoder cache commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmer.config import load_yaml
from mmer.encoders import DownloadApprovalRequired, describe_encoder


def cache_main(modality: str) -> int:
    parser = argparse.ArgumentParser(
        description=f"Inspect the configured {modality} encoder without downloading weights."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/encoder/frozen_encoders.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(describe_encoder(modality, load_yaml(args.config), args.dry_run), indent=2))
    except DownloadApprovalRequired as exc:
        print(str(exc))
        return 2
    return 0

