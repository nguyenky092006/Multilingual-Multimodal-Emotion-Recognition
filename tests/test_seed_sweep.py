from __future__ import annotations

import pytest

from mmer.experiments import aggregate_seed_metrics, seed_config


def test_seed_config_replaces_existing_seed_suffix_without_mutating_input():
    base = {
        "experiment_name": "p3_meta_seed17",
        "seed": 17,
        "output_dir": "outputs/full/p3_meta_seed17",
    }
    generated = seed_config(base, 41)
    assert generated["experiment_name"] == "p3_meta_seed41"
    assert generated["output_dir"] == "outputs/full/p3_meta_seed41"
    assert generated["seed"] == 41
    assert base["seed"] == 17


def test_seed_metric_aggregation_reports_sample_statistics():
    records = [
        {"protocol": "held_out_speaker", "training_mode": "supervised", "paper_ready": True,
         "uar": 0.4, "macro_f1": 0.3, "accuracy": 0.5},
        {"protocol": "held_out_speaker", "training_mode": "supervised", "paper_ready": True,
         "uar": 0.5, "macro_f1": 0.4, "accuracy": 0.6},
        {"protocol": "held_out_speaker", "training_mode": "supervised", "paper_ready": True,
         "uar": 0.6, "macro_f1": 0.5, "accuracy": 0.7},
    ]
    result = aggregate_seed_metrics(records, [17, 23, 41])
    assert result["paper_ready"] is True
    assert result["metrics"]["uar"]["mean"] == pytest.approx(0.5)
    assert result["metrics"]["uar"]["std"] == pytest.approx(0.1)
