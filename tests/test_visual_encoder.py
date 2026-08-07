from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open

from mmer.encoders.visual import (
    VisualCacheInput,
    VisualClip,
    cache_visual_embeddings,
    uniform_frame_indices,
)


class FakeImageProcessor:
    def __call__(self, images, return_tensors):
        assert return_tensors == "pt"
        values = torch.stack(
            [torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0 for image in images]
        )
        return {"pixel_values": values}


class FakeVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(_commit_hash="fake-visual-revision")

    def forward(self, pixel_values, **kwargs):
        del kwargs
        channels = pixel_values.mean(dim=(2, 3))
        pooled = torch.cat((channels, channels.mean(dim=1, keepdim=True)), dim=1)
        return SimpleNamespace(pooler_output=pooled)


def _fake_decoder(path: Path, frames_per_clip: int) -> VisualClip:
    offset = 10 if path.name.startswith("a") else 40
    decoded = 7
    indices = uniform_frame_indices(decoded, frames_per_clip)
    frames = tuple(
        np.full((6, 8, 3), offset + index, dtype=np.uint8)
        for index in indices
    )
    return VisualClip(
        frames=frames,
        decoded_frame_count=decoded,
        selected_indices=indices,
        width=8,
        height=6,
        fps=25.0,
        duration_seconds=decoded / 25.0,
        codec="fake",
        mean_brightness=float(np.mean([frame.mean() / 255.0 for frame in frames])),
        mean_gradient_energy=0.0,
    )


def test_uniform_frame_indices_span_clip_and_handle_short_video():
    assert uniform_frame_indices(67, 8) == (0, 9, 19, 28, 38, 47, 57, 66)
    assert uniform_frame_indices(3, 8) == (0, 1, 2)
    assert uniform_frame_indices(9, 1) == (4,)
    with pytest.raises(ValueError, match="positive"):
        uniform_frame_indices(0, 8)


def test_visual_cache_is_full_frame_safe_and_resumable(tmp_path: Path):
    manifest = tmp_path / "data" / "manifests" / "pilot.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    first_video = tmp_path / "data" / "raw" / "a.flv"
    second_video = tmp_path / "data" / "raw" / "b.flv"
    first_video.parent.mkdir(parents=True)
    first_video.write_bytes(b"fake-video-a")
    second_video.write_bytes(b"fake-video-b")
    inputs = [
        VisualCacheInput("sample-a", "data/raw/a.flv"),
        VisualCacheInput("sample-b", "data/raw/b.flv"),
    ]
    kwargs = {
        "inputs": inputs,
        "project_root": tmp_path,
        "output_dir": "data/cache/visual/test",
        "manifest_path": manifest,
        "identifier": "fake/siglip",
        "revision": "fake-visual-revision",
        "resolved_revision": "fake-visual-revision",
        "device": "cpu",
        "batch_size": 2,
        "frames_per_clip": 4,
        "inference_precision": "float32",
        "expected_embedding_dimension": 4,
        "image_processor": FakeImageProcessor(),
        "model": FakeVisionModel(),
        "frame_decoder": _fake_decoder,
    }
    first = cache_visual_embeddings(**kwargs)
    assert first.processed == 2
    assert first.skipped == 0
    assert first.embedding_dimension == 4

    cache_file = first.output_dir / "embeddings" / "sample-a.safetensors"
    with safe_open(cache_file, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor("embedding")
        metadata = handle.metadata()
    assert embedding.shape == (4,)
    assert torch.isfinite(embedding).all()
    assert metadata["valid_frame_count"] == "4"
    assert metadata["selected_frame_indices"] == "0,2,4,6"
    assert metadata["face_crop"] == "false"
    assert len(metadata["source_video_sha256"]) == 64

    second = cache_visual_embeddings(**kwargs)
    assert second.processed == 0
    assert second.skipped == 2
    rows = [
        json.loads(line)
        for line in (second.output_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in rows] == ["sample-a", "sample-b"]


def test_face_crop_is_not_silently_enabled(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    video = tmp_path / "a.flv"
    video.write_bytes(b"fake")
    with pytest.raises(NotImplementedError, match="later ablation"):
        cache_visual_embeddings(
            inputs=[VisualCacheInput("a", "a.flv")],
            project_root=tmp_path,
            output_dir="cache",
            manifest_path=manifest,
            identifier="fake/siglip",
            device="cpu",
            inference_precision="float32",
            expected_embedding_dimension=4,
            face_crop=True,
            image_processor=FakeImageProcessor(),
            model=FakeVisionModel(),
            frame_decoder=_fake_decoder,
        )


def test_visual_cache_contract_mismatch_requires_new_directory(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    video = tmp_path / "a.flv"
    video.write_bytes(b"fake")
    common = {
        "inputs": [VisualCacheInput("a", "a.flv")],
        "project_root": tmp_path,
        "output_dir": "cache",
        "manifest_path": manifest,
        "identifier": "fake/siglip",
        "revision": "one",
        "resolved_revision": "one",
        "device": "cpu",
        "inference_precision": "float32",
        "expected_embedding_dimension": 4,
        "image_processor": FakeImageProcessor(),
        "model": FakeVisionModel(),
        "frame_decoder": _fake_decoder,
    }
    cache_visual_embeddings(**common)
    with pytest.raises(RuntimeError, match="cache contract differs"):
        cache_visual_embeddings(**{**common, "frames_per_clip": 3})
