from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mmer.data import ManifestSample, validate_manifest
from mmer.data.real_cached import load_real_cache_bundle
from mmer.fusion import ConcatenationFusion, ReliabilityGatedFusion, masked_softmax
from mmer.models.trimodal import TrimodalEmotionModel, parameter_counts


DIMS = {"audio": 4, "text": 3, "visual": 2}


def _model(enabled):
    return TrimodalEmotionModel(
        input_dims=DIMS,
        num_classes=4,
        languages=["en"],
        corpora=["fixture"],
        d_model=8,
        projection_hidden=12,
        adapter_bottleneck=4,
        dropout=0.0,
        fusion="concat",
        enabled_modalities=enabled,
    )


def test_disabled_modalities_do_not_create_parameters_or_nonzero_weights():
    audio_only = _model(["audio"])
    trimodal = _model(["audio", "text", "visual"])
    assert parameter_counts(audio_only)["trainable"] < parameter_counts(trimodal)["trainable"]
    output = audio_only(
        {"audio": torch.randn(2, 4)},
        torch.ones(2, 3, dtype=torch.bool),
        ["en", "en"],
        ["fixture", "fixture"],
        torch.ones(2, 3),
    )
    assert output["effective_modality_mask"].tolist() == [[True, False, False]] * 2
    assert torch.equal(output["fusion_weights"][:, 1:], torch.zeros(2, 2))
    assert set(output["route_stats"]) == {"audio"}


def test_subset_fusion_has_fewer_parameters_and_canonical_weights():
    subset = ConcatenationFusion(8, enabled_modalities=["audio", "text"])
    full = ConcatenationFusion(8)
    assert sum(item.numel() for item in subset.parameters()) < sum(
        item.numel() for item in full.parameters()
    )
    representations = torch.randn(2, 3, 8)
    output, weights = subset(representations, torch.ones(2, 3, dtype=torch.bool))
    assert output.shape == (2, 8)
    assert torch.equal(weights[:, 2], torch.zeros(2))

    gated = ReliabilityGatedFusion(8, enabled_modalities=["audio", "text"])
    _, gated_weights = gated(representations, torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 3))
    assert torch.equal(gated_weights[:, 2], torch.zeros(2))


def test_masked_softmax_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="equal shape"):
        masked_softmax(torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bool))


def _audio_only_fixture(root: Path):
    manifest = root / "data" / "manifests" / "audio_only.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = []
    for split in ("train", "validation", "test"):
        rows.append(
            {
                "sample_id": f"{split}-1",
                "audio_path": f"data/raw/{split}-1.wav",
                "video_path": None,
                "transcript": None,
                "emotion": "angry",
                "speaker_id": f"speaker-{split}",
                "language": "en",
                "corpus": "fixture",
                "split": split,
                "duration": 1.0,
                "transcript_source": None,
                "asr_confidence": None,
                "audio_available": True,
                "text_available": False,
                "visual_available": False,
                "source_video_id": None,
            }
        )
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cache_root = root / "data" / "cache" / "audio"
    embeddings = cache_root / "embeddings"
    embeddings.mkdir(parents=True)
    contract = {
        "schema_version": 1,
        "modality": "audio",
        "model_identifier": "fixture/audio",
        "resolved_revision": "test",
        "cache_dtype": "float32",
        "embedding_dimension": 4,
        "manifest_sha256": manifest_hash,
    }
    contract_path = cache_root / "cache_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract_hash = hashlib.sha256(encoded).hexdigest()
    index_rows = []
    for row in rows:
        path = embeddings / f"{row['sample_id']}.safetensors"
        save_file(
            {"embedding": torch.ones(4)}, path, metadata={"contract_sha256": contract_hash}
        )
        index_rows.append(
            {
                "sample_id": row["sample_id"],
                "cache_path": path.relative_to(root).as_posix(),
                "embedding_dimension": 4,
                "clipping_fraction": "0.0",
            }
        )
    index_path = cache_root / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows), encoding="utf-8"
    )
    return manifest, {
        "audio": {
            "index_path": index_path.relative_to(root).as_posix(),
            "contract_path": contract_path.relative_to(root).as_posix(),
        }
    }


def test_real_bundle_requires_only_enabled_cache_and_dimension(tmp_path: Path):
    manifest, sources = _audio_only_fixture(tmp_path)
    bundle = load_real_cache_bundle(
        tmp_path,
        manifest,
        sources,
        {"angry": 0},
        {"audio": 4},
        enabled_modalities=["audio"],
    )
    assert bundle.audit["cache_contracts"].keys() == {"audio"}
    assert bundle.audit["input_dims"] == {"audio": 4, "text": 1, "visual": 1}
    assert bundle.audit["unique_cache_tensors_loaded"] == 3
    example = bundle.splits["test"][0]
    assert example.modality_mask.tolist() == [True, False, False]
    assert example.embeddings["visual"].shape == (1,)


def _manifest_sample(**updates):
    values = {
        "sample_id": "sample",
        "audio_path": None,
        "video_path": None,
        "transcript": "text",
        "emotion": "angry",
        "speaker_id": "speaker",
        "language": "en",
        "corpus": "fixture",
        "split": "train",
        "duration": 1.0,
        "transcript_source": "gold",
        "asr_confidence": None,
        "audio_available": False,
        "text_available": True,
        "visual_available": False,
    }
    values.update(updates)
    return ManifestSample(**values)


def test_manifest_rejects_all_split_speaker_leakage_and_invalid_split():
    records = [
        _manifest_sample(sample_id="a", split="train"),
        _manifest_sample(sample_id="b", split="validation", transcript="different"),
        _manifest_sample(sample_id="c", split="dev", speaker_id="other", transcript="third"),
    ]
    report = validate_manifest(records, {"angry"}, check_files=False)
    serious = {issue.code for issue in report.serious}
    assert "speaker_split_overlap" in serious
    assert "invalid_split" in serious


def test_manifest_rejects_available_modality_without_path():
    report = validate_manifest(
        [_manifest_sample(audio_available=True, audio_path=None)], {"angry"}, check_files=False
    )
    assert "availability_path_mismatch" in {issue.code for issue in report.serious}
