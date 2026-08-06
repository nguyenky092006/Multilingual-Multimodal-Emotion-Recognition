#!/usr/bin/env python
"""Extract frozen XLS-R audio embeddings from an MMER manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mmer.config import load_yaml
from mmer.data import load_manifest
from mmer.encoders import AudioCacheInput, cache_audio_embeddings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/encoder/frozen_encoders.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/cremad_pilot.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/audio/cremad_pilot_xlsr300m"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_yaml(config_path)["audio"]
    samples = load_manifest(manifest_path)
    inputs = [
        AudioCacheInput(sample.sample_id, str(sample.audio_path))
        for sample in samples
        if sample.audio_available and sample.audio_path
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        inputs = inputs[: args.limit]
    resolved = {
        "modality": "audio",
        "model_identifier": str(config["identifier"]),
        "revision": config.get("revision"),
        "manifest": manifest_path.relative_to(root).as_posix(),
        "selected_samples": len(inputs),
        "output_dir": args.output_dir.as_posix(),
        "device": args.device,
        "batch_size": args.batch_size or int(config.get("batch_size", 2)),
        "inference_precision": str(config.get("inference_precision", "float16")),
        "allow_download": args.allow_download,
    }
    if args.dry_run:
        resolved["status"] = "validated; no model loaded and no weights downloaded"
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return 0
    result = cache_audio_embeddings(
        inputs=inputs,
        project_root=root,
        output_dir=args.output_dir,
        manifest_path=manifest_path,
        identifier=str(config["identifier"]),
        revision=config.get("revision"),
        device=args.device,
        batch_size=args.batch_size or int(config.get("batch_size", 2)),
        sample_rate=int(config.get("sample_rate", 16_000)),
        max_duration_seconds=float(config.get("max_duration_seconds", 12.0)),
        inference_precision=str(config.get("inference_precision", "float16")),
        allow_download=args.allow_download,
    )
    print(json.dumps({**resolved, **asdict(result), "output_dir": str(result.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
