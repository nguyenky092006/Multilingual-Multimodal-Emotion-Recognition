from __future__ import annotations

from pathlib import Path

from mmer.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "cremad_full_baseline_audio.yaml": ["audio"],
    "cremad_full_baseline_text.yaml": ["text"],
    "cremad_full_baseline_audio_text_concat.yaml": ["audio", "text"],
}


def test_full_audio_text_configs_are_independent_and_comparable():
    shared = None
    for name, modalities in CONFIGS.items():
        config = load_yaml(ROOT / "configs" / "experiment" / name)
        assert config["pilot"] is False
        assert config["diagnostic"] is True
        assert config["seed"] == 17
        assert config["data"]["manifest_path"] == "data/manifests/cremad_full.jsonl"
        assert config["data"]["enabled_modalities"] == modalities
        assert set(config["data"]["input_dims"]) == set(modalities)
        assert set(config["data"]["caches"]) == set(modalities)
        assert config["model"]["fusion"] == "concat"
        assert config["model"]["use_modality_adapters"] is False
        assert config["model"]["emotion_adapter_mode"] == "none"
        assert config["model"]["use_routed_adapters"] is False
        comparable = {
            key: config[key]
            for key in (
                "epochs", "early_stopping_patience", "batch_size", "learning_rate",
                "weight_decay", "use_class_weights", "modality_dropout",
            )
        }
        comparable["model"] = config["model"]
        if shared is None:
            shared = comparable
        else:
            assert comparable == shared
