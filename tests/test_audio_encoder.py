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
    _duration_weighted_segment_mean,
    AudioCacheInput,
    cache_audio_embeddings,
    attentive_statistics,
    masked_mean,
    masked_statistics,
    read_pcm16_mono,
)
from mmer.config import load_yaml


def _write_wav(
    path: Path, values: np.ndarray, sample_rate: int = 16_000, channels: int = 1
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(values * 32768.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
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

    def forward(self, input_values, attention_mask, output_hidden_states=False, **kwargs):
        del attention_mask, kwargs
        hidden = torch.stack((input_values, input_values * 2.0, input_values * 3.0), dim=-1)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden * 0.5, hidden) if output_hidden_states else None,
        )


def test_read_pcm16_downmixes_and_resamples(tmp_path: Path):
    path = tmp_path / "audio.wav"
    _write_wav(path, np.array([0.0, 0.25, -0.5, 0.99999], dtype=np.float32))
    signal = read_pcm16_mono(path)
    assert signal.sample_rate == 16_000
    assert signal.duration_seconds == pytest.approx(4 / 16_000)
    assert signal.peak == pytest.approx(32767 / 32768)
    assert signal.clipping_fraction == pytest.approx(0.25)

    wrong_rate = tmp_path / "wrong.wav"
    _write_wav(wrong_rate, np.ones(20, dtype=np.float32) * 0.1, sample_rate=8_000)
    resampled = read_pcm16_mono(wrong_rate)
    assert resampled.source_sample_rate == 8_000
    assert resampled.sample_rate == 16_000
    assert len(resampled.waveform) == 40

    stereo = tmp_path / "stereo.wav"
    channels = np.stack((np.ones(8) * 0.5, np.ones(8) * -0.5), axis=1).astype(np.float32)
    _write_wav(stereo, channels, channels=2)
    downmixed = read_pcm16_mono(stereo)
    assert downmixed.source_channels == 2
    assert np.max(np.abs(downmixed.waveform)) < 1e-4


def test_masked_mean_excludes_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    assert torch.equal(masked_mean(hidden, mask), torch.tensor([[2.0, 3.0]]))
    statistics = masked_statistics(hidden, mask)
    attentive = attentive_statistics(hidden, mask)
    assert statistics.shape == attentive.shape == (1, 4)
    assert torch.isfinite(statistics).all() and torch.isfinite(attentive).all()


def test_chunk_embeddings_are_weighted_by_segment_duration():
    pooled = torch.tensor([[1.0, 2.0], [5.0, 6.0], [20.0, 30.0]])
    combined = _duration_weighted_segment_mean(
        pooled,
        segment_owners=[0, 0, 1],
        segment_lengths=[3, 1, 2],
        owner_count=2,
    )
    assert torch.equal(combined, torch.tensor([[2.0, 3.0], [20.0, 30.0]]))


def test_emotiontalk_audio_config_preserves_long_clips_by_chunking():
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / "configs" / "encoder" / "emotiontalk_xlsr_chunk12s.yaml")["audio"]
    assert config["duration_policy"] == "chunk"
    assert config["max_duration_seconds"] == 12.0
    assert config["chunk_overlap_seconds"] == 0.0
    assert config["pooling"] == "masked_mean"


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


def test_audio_chunking_hidden_layer_and_statistics_pooling(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    audio = tmp_path / "long.wav"
    _write_wav(audio, np.linspace(-0.4, 0.4, 400, dtype=np.float32))
    result = cache_audio_embeddings(
        inputs=[AudioCacheInput("sample", "long.wav")],
        project_root=tmp_path,
        output_dir="cache_stats",
        manifest_path=manifest,
        identifier="fake/xlsr",
        device="cpu",
        inference_precision="float32",
        max_duration_seconds=0.01,
        duration_policy="chunk",
        chunk_overlap_seconds=0.002,
        pooling="attentive_statistics",
        hidden_layer=0,
        feature_extractor=FakeFeatureExtractor(),
        model=FakeModel(),
        resolved_revision="fake-revision",
    )
    assert result.embedding_dimension == 6
    with safe_open(result.output_dir / "embeddings" / "sample.safetensors", framework="pt") as handle:
        assert handle.get_tensor("embedding").shape == (6,)
        assert int(handle.metadata()["chunk_count"]) > 1
