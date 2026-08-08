from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from mmer.data.real_cached import load_real_cache_bundle
from mmer.data.cached import collate_cached
from mmer.real_runner import run_cached_evaluation, run_cached_training


LABELS = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
DIMS = {"audio": 4, "text": 3, "visual": 2}


def _json_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_fixture(root: Path, speaker_leakage: bool = False):
    manifest = root / "data" / "manifests" / "pilot.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label_index, emotion in enumerate(LABELS):
            sample_id = f"{split}-{emotion}"
            speaker = f"speaker-{split}-{label_index}"
            if speaker_leakage and split == "test" and label_index == 0:
                speaker = "speaker-train-0"
            rows.append(
                {
                    "sample_id": sample_id,
                    "audio_path": f"data/raw/{sample_id}.wav",
                    "video_path": f"data/raw/{sample_id}.flv",
                    "transcript": f"Official prompt {label_index}",
                    "emotion": emotion,
                    "speaker_id": speaker,
                    "language": "en",
                    "corpus": "fixture",
                    "split": split,
                    "duration": 2.0,
                    "transcript_source": "official_audio_transcript",
                    "asr_confidence": None,
                    "audio_available": True,
                    "text_available": True,
                    "visual_available": True,
                    "source_video_id": sample_id,
                }
            )
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()

    specifications = {}
    for modality in ("audio", "text", "visual"):
        cache_root = root / "data" / "cache" / modality
        embeddings = cache_root / "embeddings"
        embeddings.mkdir(parents=True)
        contract = {
            "schema_version": 1,
            "modality": modality,
            "model_identifier": f"fixture/{modality}",
            "resolved_revision": f"{modality}-revision",
            "cache_dtype": "float32",
            "embedding_dimension": DIMS[modality],
        }
        if modality != "text":
            contract["manifest_sha256"] = manifest_hash
        if modality == "visual":
            contract["frames_per_clip"] = 8
        contract_path = cache_root / "cache_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        contract_hash = _json_hash(contract)
        index_rows = []
        for sample_index, row in enumerate(rows):
            path = embeddings / f"{row['sample_id']}.safetensors"
            vector = torch.arange(DIMS[modality], dtype=torch.float32) + sample_index / 10
            vector[LABELS[row["emotion"]] % DIMS[modality]] += 2.0
            save_file(
                {"embedding": vector},
                path,
                metadata={"contract_sha256": contract_hash},
            )
            index_row = {
                "sample_id": row["sample_id"],
                "cache_path": path.relative_to(root).as_posix(),
                "embedding_dimension": DIMS[modality],
            }
            if modality == "audio":
                index_row["clipping_fraction"] = "0.1"
            if modality == "visual":
                index_row["valid_frame_count"] = "6"
            index_rows.append(index_row)
        index_path = cache_root / "index.jsonl"
        index_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
            encoding="utf-8",
        )
        spec = {
            "index_path": index_path.relative_to(root).as_posix(),
            "contract_path": contract_path.relative_to(root).as_posix(),
        }
        if modality == "text":
            summary_path = cache_root / "summary.json"
            summary_path.write_text(
                json.dumps({"manifest_sha256": manifest_hash}), encoding="utf-8"
            )
            spec["summary_path"] = summary_path.relative_to(root).as_posix()
        specifications[modality] = spec
    return manifest, specifications


def test_real_cache_bundle_aligns_splits_dimensions_and_quality(tmp_path: Path):
    manifest, sources = _write_fixture(tmp_path)
    bundle = load_real_cache_bundle(
        tmp_path,
        manifest,
        sources,
        LABELS,
        DIMS,
    )
    assert bundle.audit["split_counts"] == {"test": 4, "train": 4, "validation": 4}
    assert bundle.audit["split_corpus_counts"] == {
        "test": {"fixture": 4},
        "train": {"fixture": 4},
        "validation": {"fixture": 4},
    }
    assert bundle.audit["unique_cache_tensors_loaded"] == 36
    example = bundle.splits["train"][0]
    assert example.modality_mask.tolist() == [True, True, True]
    assert example.quality.tolist() == pytest.approx([0.9, 1.0, 0.75])
    assert example.embeddings["audio"].shape == (4,)
    assert example.embeddings["text"].shape == (3,)
    assert example.embeddings["visual"].shape == (2,)


def test_real_cache_bundle_rejects_index_misalignment(tmp_path: Path):
    manifest, sources = _write_fixture(tmp_path)
    audio_index = tmp_path / sources["audio"]["index_path"]
    audio_index.write_text(
        "\n".join(audio_index.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index/manifest sample mismatch"):
        load_real_cache_bundle(tmp_path, manifest, sources, LABELS, DIMS)


def test_real_cache_bundle_rejects_speaker_leakage(tmp_path: Path):
    manifest, sources = _write_fixture(tmp_path, speaker_leakage=True)
    with pytest.raises(ValueError, match="speaker leakage"):
        load_real_cache_bundle(tmp_path, manifest, sources, LABELS, DIMS)


def test_real_cache_bundle_loads_frame_level_visual_tensor(tmp_path: Path):
    manifest, sources = _write_fixture(tmp_path)
    visual_contract_path = tmp_path / sources["visual"]["contract_path"]
    contract = json.loads(visual_contract_path.read_text(encoding="utf-8"))
    contract["stores_frame_embeddings"] = True
    contract["temporal_pooling"] = "deferred_attention"
    visual_contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    contract_hash = _json_hash(contract)
    visual_index = tmp_path / sources["visual"]["index_path"]
    for row in [json.loads(line) for line in visual_index.read_text().splitlines()]:
        path = tmp_path / row["cache_path"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            embedding = handle.get_tensor("embedding").clone()
        save_file(
            {
                "embedding": embedding,
                "frame_embeddings": torch.stack([embedding, embedding + 0.1]),
            },
            path,
            metadata={"contract_sha256": contract_hash},
        )
    sources["visual"]["tensor_key"] = "frame_embeddings"
    bundle = load_real_cache_bundle(tmp_path, manifest, sources, LABELS, DIMS)
    example = bundle.splits["train"][0]
    assert example.embeddings["visual"].shape == (2, DIMS["visual"])
    batch = collate_cached(bundle.splits["train"].examples)
    assert batch["temporal_masks"]["visual"].shape == (4, 2)
    assert bundle.audit["cache_contracts"]["visual"]["tensor_key"] == "frame_embeddings"


def test_real_cached_runner_saves_and_reloads_checkpoint(tmp_path: Path):
    manifest, sources = _write_fixture(tmp_path)
    labels_path = tmp_path / "configs" / "data" / "labels.yaml"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text(yaml.safe_dump({"labels": LABELS}), encoding="utf-8")
    config = {
        "experiment_name": "fixture-real-cache",
        "pilot": True,
        "seed": 17,
        "device": "cpu",
        "epochs": 1,
        "early_stopping_patience": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "use_class_weights": True,
        "modality_dropout": 0.1,
        "output_dir": "outputs/fixture",
        "labels_path": labels_path.relative_to(tmp_path).as_posix(),
        "data": {
            "synthetic": False,
            "manifest_path": manifest.relative_to(tmp_path).as_posix(),
            "enabled_modalities": ["audio", "text", "visual"],
            "quality_policy": "validated_metadata_v1",
            "input_dims": DIMS,
            "caches": sources,
        },
        "model": {
            "d_model": 8,
            "projection_hidden": 12,
            "adapter_bottleneck": 4,
            "dropout": 0.0,
            "fusion": "reliability",
            "shared_adapter": True,
            "routing_alpha": 0.5,
            "languages": ["en"],
            "corpora": ["fixture"],
        },
    }
    config_path = tmp_path / "configs" / "experiment" / "fixture.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = run_cached_training(config_path, tmp_path)
    assert summary["synthetic"] is False
    assert summary["training_mode"] == "supervised"
    assert summary["protocol"] == "speaker_disjoint_full_test"
    assert summary["pilot"] is True
    assert summary["epochs_completed"] == 1
    assert len(summary["source_snapshot_sha256"]) == 64
    assert "diagnostic_flags" in summary
    assert (tmp_path / summary["checkpoint_path"]).is_file()
    metrics = run_cached_evaluation(
        config_path,
        project_root=tmp_path,
        modality_subsets=[["audio"], ["text", "visual"]],
    )
    assert metrics["synthetic"] is False
    assert metrics["training_mode"] == "supervised"
    assert metrics["protocol"] == "speaker_disjoint_full_test"
    assert metrics["pilot"] is True
    assert metrics["checkpoint_epoch"] == 1
    assert metrics["source_snapshot_matches_checkpoint"] is True
    assert "group_metrics" in metrics
    assert "transcript_source" in metrics["group_metrics"]
    assert set(metrics["modality_stress"]) == {"audio", "text+visual"}
    assert metrics["modality_stress"]["audio"]["forced_modality_subset"] == ["audio"]
