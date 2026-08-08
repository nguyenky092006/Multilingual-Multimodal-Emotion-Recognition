from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_results.py"
SPEC = importlib.util.spec_from_file_location("mmer_aggregate_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(score: float, protocol: str = "speaker_disjoint_5_shot_episode_mean"):
    metric = {"mean": score, "std": 0.0, "ci95_low": score, "ci95_high": score}
    return {
        "synthetic": False,
        "paper_ready": False,
        "training_mode": "episodic",
        "protocol": protocol,
        "uar": score,
        "macro_f1": score,
        "accuracy": score,
        "few_shot": {"1": {name: metric for name in MODULE.METRICS}},
    }


def test_aggregate_reports_across_run_confidence_intervals():
    result = MODULE.aggregate([_record(0.4), _record(0.6)], ["a.json", "b.json"])
    assert result["runs"] == 2
    assert result["metrics"]["uar"]["mean"] == pytest.approx(0.5)
    assert result["metrics"]["uar"]["std"] > 0.0
    assert result["few_shot_across_runs"]["1"]["macro_f1"]["mean"] == pytest.approx(0.5)
    assert result["paper_ready"] is False


def test_aggregate_refuses_mixed_protocols():
    with pytest.raises(ValueError, match="refusing to mix"):
        MODULE.aggregate([_record(0.4), _record(0.6, "zero_shot")], ["a", "b"])


def test_aggregate_accepts_explicit_protocol_for_legacy_supervised_records():
    records = [
        {"uar": 0.4, "macro_f1": 0.3, "accuracy": 0.5},
        {"uar": 0.6, "macro_f1": 0.5, "accuracy": 0.7},
    ]
    result = MODULE.aggregate(
        records,
        ["legacy-a.json", "legacy-b.json"],
        protocol_override="speaker_disjoint_full_test",
    )
    assert result["protocol"] == "speaker_disjoint_full_test"
    assert result["training_mode"] == "supervised"
    assert result["protocol_override_applied"] is True


def test_aggregate_refuses_protocol_override_conflicting_with_explicit_record():
    with pytest.raises(ValueError, match="conflicts with explicit protocols"):
        MODULE.aggregate(
            [_record(0.4)],
            ["explicit.json"],
            protocol_override="speaker_disjoint_full_test",
        )
