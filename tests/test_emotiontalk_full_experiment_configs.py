from __future__ import annotations

from pathlib import Path

from mmer.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = (
    "emotiontalk_full_baseline_audio.yaml",
    "emotiontalk_full_baseline_text.yaml",
    "emotiontalk_full_baseline_audio_text_concat.yaml",
    "emotiontalk_full_p1_reliability.yaml",
    "emotiontalk_full_p2_meta.yaml",
    "emotiontalk_full_p3_meta.yaml",
)


def test_emotiontalk_full_configs_bind_verified_full_caches_without_visual():
    for name in CONFIG_NAMES:
        config = load_yaml(ROOT / "configs" / "experiment" / name)
        data = config["data"]
        model = config["model"]
        enabled = set(data["enabled_modalities"])
        assert data["manifest_path"] == "data/manifests/emotiontalk_full.jsonl"
        assert enabled and enabled <= {"audio", "text"}
        assert set(data["input_dims"]) == enabled
        assert set(data["caches"]) == enabled
        assert model["languages"] == ["zh"]
        assert model["corpora"] == ["emotiontalk"]
        assert config["pilot"] is False and config["diagnostic"] is True
        assert config["use_class_weights"] is True
        if "audio" in enabled:
            assert "emotiontalk_full_xlsr300m_chunk12s" in data["caches"]["audio"]["index_path"]
        if "text" in enabled:
            assert "emotiontalk_full_qwen3_embedding_0.6b" in data["caches"]["text"]["index_path"]


def test_emotiontalk_full_p2_p3_difference_is_supervised_contrastive_loss():
    p2 = load_yaml(ROOT / "configs" / "experiment" / "emotiontalk_full_p2_meta.yaml")
    p3 = load_yaml(ROOT / "configs" / "experiment" / "emotiontalk_full_p3_meta.yaml")
    assert p2["meta"]["enabled"] is True and p3["meta"]["enabled"] is True
    assert p2["meta"]["lambda_supcon"] == 0.0
    assert p3["meta"]["lambda_supcon"] == 0.1
    for config in (p2, p3):
        assert config["meta"]["disjoint_speakers_validation"] is True
        assert config["meta"]["disjoint_speakers_test"] is True
        assert config["meta"]["evaluation_k_shots"] == [1, 5, 10]
