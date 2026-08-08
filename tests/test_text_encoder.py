from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from mmer.encoders.text import (
    TextCacheInput,
    cache_text_embeddings,
    last_token_pool,
    masked_mean_pool,
    prompt_label_audit,
)


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "right"

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding and truncation and return_tensors == "pt"
        sequences = [[(ord(character) % 61) + 1 for character in text][:max_length] for text in texts]
        longest = max(len(sequence) for sequence in sequences)
        input_ids = torch.zeros((len(sequences), longest), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            values = torch.tensor(sequence, dtype=torch.long)
            if self.padding_side == "left":
                input_ids[index, -len(sequence) :] = values
                attention_mask[index, -len(sequence) :] = 1
            else:
                input_ids[index, : len(sequence)] = values
                attention_mask[index, : len(sequence)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(_commit_hash="fake-text-revision")

    def forward(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        values = input_ids.float()
        hidden = torch.stack((values, values.square(), values * 0.5, values + 3.0), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_last_token_pool_supports_left_and_right_padding():
    hidden = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],
            [[3.0], [4.0], [0.0]],
        ]
    )
    right_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    assert torch.equal(last_token_pool(hidden, right_mask), torch.tensor([[2.0], [4.0]]))

    left_hidden = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],
            [[0.0], [3.0], [4.0]],
        ]
    )
    left_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    assert torch.equal(last_token_pool(left_hidden, left_mask), torch.tensor([[2.0], [4.0]]))


def test_masked_mean_pool_excludes_padding_tokens():
    hidden = torch.tensor([[[2.0], [4.0], [100.0]], [[3.0], [9.0], [15.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert torch.equal(masked_mean_pool(hidden, mask), torch.tensor([[3.0], [9.0]]))


def test_text_cache_deduplicates_maps_and_resumes(tmp_path: Path):
    manifest = tmp_path / "data" / "manifests" / "pilot.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    inputs = [
        TextCacheInput("a", "Repeated sentence.", "angry", "train", "AAA"),
        TextCacheInput("b", "Repeated sentence.", "happy", "validation", "AAA"),
        TextCacheInput("c", "Different sentence.", "sad", "test", "BBB"),
    ]
    kwargs = {
        "inputs": inputs,
        "project_root": tmp_path,
        "output_dir": "data/cache/text/fake",
        "manifest_path": manifest,
        "identifier": "fake/qwen",
        "revision": "fake-text-revision",
        "resolved_revision": "fake-text-revision",
        "device": "cpu",
        "batch_size": 2,
        "max_length": 64,
        "inference_precision": "float32",
        "expected_embedding_dimension": 4,
        "tokenizer": FakeTokenizer(),
        "model": FakeModel(),
    }
    first = cache_text_embeddings(**kwargs)
    assert first.selected_samples == 3
    assert first.unique_transcripts == 2
    assert first.processed_unique == 2
    assert first.skipped_unique == 0
    assert len(list((first.output_dir / "embeddings").glob("*.safetensors"))) == 2

    rows = [json.loads(line) for line in first.index_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["cache_path"] == rows[1]["cache_path"]
    assert rows[0]["cache_path"] != rows[2]["cache_path"]
    with safe_open(tmp_path / rows[0]["cache_path"], framework="pt", device="cpu") as handle:
        vector = handle.get_tensor("embedding")
    assert vector.shape == (4,)
    assert vector.norm().item() == pytest.approx(1.0)

    second = cache_text_embeddings(**kwargs)
    assert second.processed_unique == 0
    assert second.skipped_unique == 2


def test_shared_embeddings_support_a_second_manifest(tmp_path: Path):
    first_manifest = tmp_path / "pilot.jsonl"
    second_manifest = tmp_path / "full.jsonl"
    first_manifest.write_text("pilot\n", encoding="utf-8")
    second_manifest.write_text("full\n", encoding="utf-8")
    common = {
        "project_root": tmp_path,
        "output_dir": "cache",
        "identifier": "fake/qwen",
        "revision": "revision",
        "resolved_revision": "revision",
        "device": "cpu",
        "inference_precision": "float32",
        "expected_embedding_dimension": 4,
        "tokenizer": FakeTokenizer(),
        "model": FakeModel(),
    }
    cache_text_embeddings(
        inputs=[TextCacheInput("pilot-a", "Same text")],
        manifest_path=first_manifest,
        **common,
    )
    result = cache_text_embeddings(
        inputs=[TextCacheInput("full-a", "Same text"), TextCacheInput("full-b", "Same text")],
        manifest_path=second_manifest,
        **common,
    )
    assert result.processed_unique == 0
    assert result.skipped_unique == 1
    assert result.index_path.name == "full.jsonl"
    assert len(result.index_path.read_text(encoding="utf-8").splitlines()) == 2


def test_prompt_label_audit_exposes_frequency_artifact():
    inputs = [
        TextCacheInput("1", "Prompt A", "angry", "train", "AAA"),
        TextCacheInput("2", "Prompt A", "angry", "test", "AAA"),
        TextCacheInput("3", "Prompt B", "neutral", "train", "BBB"),
        TextCacheInput("4", "Prompt B", "neutral", "test", "BBB"),
    ]
    audit = prompt_label_audit(inputs)
    assert audit["unique_transcripts"] == 2
    assert audit["prompts_spanning_multiple_splits"] == 2
    assert audit["cramers_v_prompt_vs_emotion"] == pytest.approx(1.0)
    assert audit["mutual_information_bits"] == pytest.approx(1.0)
