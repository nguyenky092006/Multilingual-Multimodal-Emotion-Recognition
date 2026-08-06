from __future__ import annotations

import json

import pytest

from mmer.data import ManifestSample, load_manifest, validate_manifest, write_manifest

LABELS = {"angry", "happy", "neutral", "sad"}


def sample(sample_id: str = "s1", **updates: object) -> ManifestSample:
    values = {
        "sample_id": sample_id,
        "audio_path": None,
        "video_path": None,
        "transcript": "A unique utterance",
        "emotion": "happy",
        "speaker_id": "speaker-1",
        "language": "en",
        "corpus": "corpus-a",
        "split": "train",
        "duration": 1.2,
        "transcript_source": "gold",
        "asr_confidence": None,
        "audio_available": False,
        "text_available": True,
        "visual_available": False,
        "country": None,
        "region": None,
        "accent": None,
        "source_video_id": None,
    }
    values.update(updates)
    return ManifestSample(**values)


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    expected = [sample()]
    write_manifest(expected, path)
    assert load_manifest(path) == expected
    assert load_manifest(path)[0].modality_mask() == (False, True, False)


def test_missing_required_manifest_field_rejected(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"sample_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing manifest fields"):
        load_manifest(path)


def test_unsupported_emotion_is_serious():
    report = validate_manifest([sample(emotion="surprise")], LABELS, check_files=False)
    assert not report.is_valid
    assert {issue.code for issue in report.serious} == {"unsupported_label"}


def test_duplicate_sample_id_is_rejected():
    report = validate_manifest([sample(), sample()], LABELS, check_files=False)
    assert "duplicate_sample_id" in {issue.code for issue in report.serious}


def test_speaker_train_test_leakage_is_rejected():
    records = [sample("train", split="train"), sample("test", split="test", transcript="different words")]
    report = validate_manifest(records, LABELS, check_files=False)
    assert "speaker_train_test_overlap" in {issue.code for issue in report.serious}
    with pytest.raises(RuntimeError, match="speaker_train_test_overlap"):
        report.raise_for_errors()


def test_source_video_split_leakage_is_rejected():
    records = [
        sample("a", source_video_id="movie-1", split="train"),
        sample("b", source_video_id="movie-1", split="validation", speaker_id="speaker-2", transcript="other"),
    ]
    report = validate_manifest(records, LABELS, check_files=False)
    assert "source_video_split_overlap" in {issue.code for issue in report.serious}


def test_language_corpus_confounding_is_reported():
    records = [sample("a"), sample("b", language="zh", corpus="corpus-b", speaker_id="speaker-2")]
    report = validate_manifest(records, LABELS, check_files=False)
    assert "language_corpus_confounding" in {issue.code for issue in report.issues}


def test_all_missing_modalities_rejected():
    record = sample(transcript=None, text_available=False)
    report = validate_manifest([record], LABELS, check_files=False)
    assert "no_available_modality" in {issue.code for issue in report.serious}

