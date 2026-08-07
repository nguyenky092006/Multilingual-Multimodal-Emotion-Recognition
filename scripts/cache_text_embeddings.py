#!/usr/bin/env python
"""Extract deduplicated Qwen3 text embeddings from an MMER manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mmer.config import load_yaml
from mmer.data import load_manifest
from mmer.encoders import TextCacheInput, cache_text_embeddings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/encoder/frozen_encoders.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/cremad_pilot.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/text/qwen3_embedding_0.6b"))
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
    config = load_yaml(config_path)["text"]
    samples = load_manifest(manifest_path)
    inputs = [
        TextCacheInput(
            sample_id=sample.sample_id,
            transcript=str(sample.transcript),
            emotion=sample.emotion,
            split=sample.split,
            sentence_code=sample.sentence_code,
        )
        for sample in samples
        if sample.text_available and sample.transcript
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        inputs = inputs[: args.limit]
    unique_transcripts = len({item.transcript for item in inputs})
    resolved = {
        "modality": "text",
        "model_identifier": str(config["identifier"]),
        "revision": config.get("revision"),
        "manifest": manifest_path.relative_to(root).as_posix(),
        "selected_samples": len(inputs),
        "unique_transcripts": unique_transcripts,
        "deduplicated_encoder_calls_saved": len(inputs) - unique_transcripts,
        "output_dir": args.output_dir.as_posix(),
        "device": args.device,
        "batch_size": args.batch_size or int(config.get("batch_size", 16)),
        "inference_precision": str(config.get("inference_precision", "bfloat16")),
        "allow_download": args.allow_download,
    }
    if args.dry_run:
        resolved["status"] = "validated; no model loaded and no weights downloaded"
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return 0
    result = cache_text_embeddings(
        inputs=inputs,
        project_root=root,
        output_dir=args.output_dir,
        manifest_path=manifest_path,
        identifier=str(config["identifier"]),
        revision=config.get("revision"),
        device=args.device,
        batch_size=args.batch_size or int(config.get("batch_size", 16)),
        max_length=int(config.get("max_length", 128)),
        inference_precision=str(config.get("inference_precision", "bfloat16")),
        normalize_embeddings=bool(config.get("normalize_embeddings", True)),
        instruction=config.get("instruction"),
        expected_embedding_dimension=int(config.get("embedding_dimension", 1024)),
        allow_download=args.allow_download,
    )
    payload = {**resolved, **asdict(result)}
    payload["output_dir"] = str(result.output_dir)
    payload["index_path"] = str(result.index_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
