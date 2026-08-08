#!/usr/bin/env python
"""Extract frozen SigLIP visual embeddings from an MMER manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mmer.config import load_yaml
from mmer.data import load_manifest
from mmer.encoders import VisualCacheInput, cache_visual_embeddings, decode_uniform_frames


def _decode_check(inputs: list[VisualCacheInput], root: Path, frames_per_clip: int) -> dict:
    records: list[dict] = []
    for item in inputs:
        path = (root / item.video_path).resolve()
        clip = decode_uniform_frames(path, frames_per_clip)
        records.append(
            {
                "sample_id": item.sample_id,
                "decoded_frames": clip.decoded_frame_count,
                "selected_frames": len(clip.frames),
                "resolution": f"{clip.width}x{clip.height}",
                "codec": clip.codec,
                "brightness": clip.mean_brightness,
                "gradient_energy": clip.mean_gradient_energy,
            }
        )
    return {
        "status": "decoded successfully; no model loaded and no weights downloaded",
        "checked_clips": len(records),
        "decoded_frame_count_min": min(row["decoded_frames"] for row in records),
        "decoded_frame_count_max": max(row["decoded_frames"] for row in records),
        "selected_frame_count_min": min(row["selected_frames"] for row in records),
        "selected_frame_count_max": max(row["selected_frames"] for row in records),
        "mean_brightness": float(np.mean([row["brightness"] for row in records])),
        "mean_gradient_energy": float(np.mean([row["gradient_energy"] for row in records])),
        "resolutions": dict(Counter(row["resolution"] for row in records)),
        "codecs": dict(Counter(row["codec"] for row in records)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/encoder/frozen_encoders.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/cremad_pilot.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache/visual/cremad_pilot_siglip_base_p16_224"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decode-check", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_yaml(config_path)["visual"]
    samples = load_manifest(manifest_path)
    inputs = [
        VisualCacheInput(sample.sample_id, str(sample.video_path))
        for sample in samples
        if sample.visual_available and sample.video_path
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        inputs = inputs[: args.limit]
    if not inputs:
        raise ValueError("manifest selection contains no available visual clips")

    frames_per_clip = int(config.get("frames_per_clip", 8))
    temporal_pooling = str(config.get("temporal_pooling", config.get("pooling", "mean")))
    resolved = {
        "modality": "visual",
        "model_identifier": str(config["identifier"]),
        "revision": config.get("revision"),
        "manifest": manifest_path.relative_to(root).as_posix(),
        "selected_samples": len(inputs),
        "output_dir": args.output_dir.as_posix(),
        "device": args.device,
        "batch_size": args.batch_size or int(config.get("batch_size", 2)),
        "frames_per_clip": frames_per_clip,
        "frame_sampling": str(config.get("frame_sampling", "uniform")),
        "face_crop": bool(config.get("face_crop", False)),
        "face_crop_backend": str(config.get("face_crop_backend", "opencv_haar")),
        "temporal_pooling": temporal_pooling,
        "inference_precision": str(config.get("inference_precision", "float16")),
        "allow_download": args.allow_download,
    }
    if args.dry_run:
        resolved["status"] = "validated; no video decoded, no model loaded, no weights downloaded"
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return 0
    if args.decode_check:
        payload = {**resolved, **_decode_check(inputs, root, frames_per_clip)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    result = cache_visual_embeddings(
        inputs=inputs,
        project_root=root,
        output_dir=args.output_dir,
        manifest_path=manifest_path,
        identifier=str(config["identifier"]),
        revision=config.get("revision"),
        device=args.device,
        batch_size=args.batch_size or int(config.get("batch_size", 2)),
        frames_per_clip=frames_per_clip,
        inference_precision=str(config.get("inference_precision", "float16")),
        expected_embedding_dimension=int(config.get("embedding_dimension", 768)),
        face_crop=bool(config.get("face_crop", False)),
        face_crop_backend=str(config.get("face_crop_backend", "opencv_haar")),
        temporal_pooling=temporal_pooling,
        allow_download=args.allow_download,
    )
    payload = {**resolved, **asdict(result)}
    payload["output_dir"] = str(result.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
