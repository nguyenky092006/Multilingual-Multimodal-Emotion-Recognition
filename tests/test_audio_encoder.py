from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open

from mmer.encoders.audio import (
    AudioCacheInput,
    cache_audio_embeddings,
    masked_mean,
    read_pcm16_mono,
)


def _write_wav(path: Path, values: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(values * 32768.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


class FakeFeatureExtractor:
    def __call__(self, waveforms, **kwargs):
        del kwargs
        longest = max(len(value) for value in waveforms)
        values = torch.zeros((len(waveforms), longest), dtype=torch.float32)
        mask = torch.zeros((len(waveforms), longest), dtype=torch.long)
        for index, waveform in enumerate(waveforms):
            values[index, : len(waveform)] = torch.from_numpy(waveform)
            mask[index, : len(waveform)] = 1
        return {"input_values": values, "attention_mask": mask}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(_commit_hash="fake-revision")

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        assert length == attention_mask.shape[1]
        return attention_mask.bool()

    def forward(self, input_values, attention_mask, **kwargs):
        del attention_mask, kwargs
        hidden = torch.stack((input_values, input_values * 2.0, input_values * 3.0), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_read_pcm16_mono_and_reject_wrong_rate(tmp_path: Path):
    path = tmp_path / "audio.wav"
    _write_wav(path, np.array([0.0, 0.25, -0.5, 0.99999], dtype=np.float32))
    signal = read_pcm16_mono(path)
    assert signal.sample_rate == 16_000
    assert signal.duration_seconds == pytest.approx(4 / 16_000)
    assert signal.peak == pytest.approx(32767 / 32768)
    assert signal.clipping_fraction == pytest.approx(0.25)

    wrong_rate = tmp_path / "wrong.wav"
    _write_wav(wrong_rate, np.ones(20, dtype=np.float32) * 0.1, sample_rate=8_000)
    with pytest.raises(ValueError, match="sample rate"):
        read_pcm16_mono(wrong_rate)


def test_masked_mean_excludes_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    assert torch.equal(masked_mean(hidden, mask), torch.tensor([[2.0, 3.0]]))


def test_audio_cache_is_safe_and_resumable(tmp_path: Path):
    manifest = tmp_path / "data" / "manifests" / "pilot.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    first_path = tmp_path / "data" / "raw" / "a.wav"
    second_path = tmp_path / "data" / "raw" / "b.wav"
    _write_wav(first_path, np.linspace(-0.5, 0.5, 320, dtype=np.float32))
    _write_wav(second_path, np.linspace(-0.25, 0.25, 240, dtype=np.float32))
    inputs = [
        AudioCacheInput("sample-a", "data/raw/a.wav"),
        AudioCacheInput("sample-b", "data/raw/b.wav"),
    ]
    kwargs = {
        "inputs": inputs,
        "project_root": tmp_path,
        "output_dir": "data/cache/audio/test",
        "manifest_path": manifest,
        "identifier": "fake/xlsr",
        "revision": "fake-revision",
        "resolved_revision": "fake-revision",
        "device": "cpu",
        "batch_size": 2,
        "inference_precision": "float32",
        "feature_extractor": FakeFeatureExtractor(),
        "model": FakeModel(),
    }
    first = cache_audio_embeddings(**kwargs)
    assert first.processed == 2
    assert first.skipped == 0
    assert first.embedding_dimension == 3

    cache_file = tmp_path / "data" / "cache" / "audio" / "test" / "embeddings" / "sample-a.safetensors"
    with safe_open(cache_file, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor("embedding")
        metadata = handle.metadata()
    assert embedding.shape == (3,)
    assert torch.isfinite(embedding).all()
    assert metadata["sample_id"] == "sample-a"
    assert len(metadata["source_audio_sha256"]) == 64

    second = cache_audio_embeddings(**kwargs)
    assert second.processed == 0
    assert second.skipped == 2
    index_rows = [
        json.loads(line)
        for line in (second.output_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in index_rows] == ["sample-a", "sample-b"]


def test_cache_contract_mismatch_requires_new_directory(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    audio = tmp_path / "audio.wav"
    _write_wav(audio, np.ones(300, dtype=np.float32) * 0.1)
    common = {
        "inputs": [AudioCacheInput("sample", "audio.wav")],
        "project_root": tmp_path,
        "output_dir": "cache",
        "manifest_path": manifest,
        "identifier": "fake/xlsr",
        "revision": "one",
        "resolved_revision": "one",
        "device": "cpu",
        "inference_precision": "float32",
        "feature_extractor": FakeFeatureExtractor(),
        "model": FakeModel(),
    }
    cache_audio_embeddings(**common)
    with pytest.raises(RuntimeError, match="cache contract differs"):
        cache_audio_embeddings(**{**common, "revision": "two", "resolved_revision": "two"})
