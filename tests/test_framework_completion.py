from __future__ import annotations

import importlib.util
from pathlib import Path

from mmer.config import load_label_mapping, load_yaml
from mmer.models import TrimodalEmotionModel
from mmer.models.trimodal import parameter_counts
from mmer.runner import _model_kwargs


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_baseline_matrix_has_executable_configs():
    registry = load_yaml(ROOT / "configs" / "baseline" / "baselines.yaml")
    expected = {
        "cremad_pilot_b1_audio.yaml": ["audio"],
        "cremad_pilot_b1_text.yaml": ["text"],
        "cremad_pilot_b1_visual.yaml": ["visual"],
        "cremad_pilot_baseline_audio_text_concat.yaml": ["audio", "text"],
        "cremad_pilot_b2_audio_visual.yaml": ["audio", "visual"],
        "cremad_pilot_b2_text_visual.yaml": ["text", "visual"],
        "cremad_pilot_b3_trimodal_concat.yaml": ["audio", "text", "visual"],
    }
    for filename, modalities in expected.items():
        config = load_yaml(ROOT / "configs" / "experiment" / filename)
        assert config["data"]["enabled_modalities"] == modalities
        assert config["model"]["use_modality_adapters"] is False
        assert config["model"]["emotion_adapter_mode"] == "none"
        assert config["model"]["use_routed_adapters"] is False
    assert registry["tracks"]["diagnostics"]


def test_b4_is_metadata_conditioned_and_b5_matches_p1_parameter_budget():
    labels = load_label_mapping(ROOT / "configs" / "data" / "labels.yaml")
    p1 = load_yaml(ROOT / "configs" / "experiment" / "cremad_pilot_supervised_reliability.yaml")
    b4 = load_yaml(ROOT / "configs" / "experiment" / "cremad_pilot_b4_metadata_embeddings.yaml")
    b5 = load_yaml(
        ROOT / "configs" / "experiment" / "cremad_pilot_b5_parameter_matched_shared.yaml"
    )
    assert b4["model"]["use_metadata_embeddings"] is True
    assert b4["model"]["use_routed_adapters"] is False
    left = TrimodalEmotionModel(**_model_kwargs(p1, len(labels)))
    right = TrimodalEmotionModel(**_model_kwargs(b5, len(labels)))
    left_count = int(parameter_counts(left)["trainable"])
    right_count = int(parameter_counts(right)["trainable"])
    assert abs(left_count - right_count) / left_count < 0.001


def test_ablation_registry_covers_required_software_ablations():
    matrix = load_yaml(ROOT / "configs" / "ablation" / "ablations.yaml")
    names = set(matrix["ablations"])
    assert {
        "no_audio",
        "no_text",
        "no_visual",
        "no_modality_adapters",
        "no_shared_adapter",
        "no_language_adapter",
        "no_corpus_adapter",
        "no_reliability_gate",
        "no_meta_learning",
        "parameter_matched_shared",
        "temporal_attention",
        "face_crop",
        "gold_vs_asr",
        "missing_modality_stress",
    } <= names
    assert matrix["ablations"]["gold_vs_asr"]["status"].startswith("blocked")
    runner = _script("run_ablation.py")
    merged = runner.deep_merge(
        {"data": {"enabled_modalities": ["audio", "text", "visual"]}},
        {"data": {"enabled_modalities": ["audio"]}},
    )
    assert merged["data"]["enabled_modalities"] == ["audio"]


def test_temporal_and_face_crop_configs_use_separate_cache_contracts():
    temporal = load_yaml(
        ROOT / "configs" / "experiment" / "cremad_pilot_ablation_temporal_attention.yaml"
    )
    visual = temporal["data"]["caches"]["visual"]
    assert visual["tensor_key"] == "frame_embeddings"
    assert temporal["model"]["visual_temporal_pooling"] == "attention"
    face = load_yaml(
        ROOT / "configs" / "experiment" / "cremad_pilot_ablation_face_crop.yaml"
    )
    assert "face_crop" in face["data"]["caches"]["visual"]["index_path"]


def test_missing_modality_scenarios_are_deduplicated():
    module = _script("stress_test_modalities.py")
    assert module._default_scenarios(["audio", "text"]) == [["audio"], ["text"]]
    scenarios = module._default_scenarios(["audio", "text", "visual"])
    assert len(scenarios) == 6
    assert ["audio", "text", "visual"] not in scenarios
