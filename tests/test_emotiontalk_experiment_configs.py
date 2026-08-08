from __future__ import annotations

from pathlib import Path

from mmer.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = (
    "emotiontalk_pilot_baseline_audio.yaml",
    "emotiontalk_pilot_baseline_text.yaml",
    "emotiontalk_pilot_baseline_audio_text_concat.yaml",
    "emotiontalk_pilot_p1_reliability.yaml",
)


def test_emotiontalk_pilot_configs_use_only_verified_audio_text_caches():
    for name in CONFIG_NAMES:
        config = load_yaml(ROOT / "configs" / "experiment" / name)
        data = config["data"]
        model = config["model"]
        enabled = set(data["enabled_modalities"])
        assert data["manifest_path"] == "data/manifests/emotiontalk_pilot.jsonl"
        assert enabled and enabled <= {"audio", "text"}
        assert set(data["input_dims"]) == enabled
        assert set(data["caches"]) == enabled
        assert model["languages"] == ["zh"]
        assert model["corpora"] == ["emotiontalk"]
        assert config["pilot"] is True and config["diagnostic"] is True
        assert "visual" not in data["caches"]
        if "audio" in enabled:
            assert "chunk12s" in data["caches"]["audio"]["contract_path"]


def test_emotiontalk_concat_and_p1_have_distinct_fusion_contracts():
    concat = load_yaml(
        ROOT / "configs" / "experiment" / "emotiontalk_pilot_baseline_audio_text_concat.yaml"
    )
    p1 = load_yaml(
        ROOT / "configs" / "experiment" / "emotiontalk_pilot_p1_reliability.yaml"
    )
    assert concat["model"]["fusion"] == "concat"
    assert concat["model"]["use_modality_adapters"] is False
    assert concat["model"]["emotion_adapter_mode"] == "none"
    assert p1["model"]["fusion"] == "reliability"
    assert p1["model"]["use_modality_adapters"] is True
    assert p1["model"]["emotion_adapter_mode"] == "shared"
