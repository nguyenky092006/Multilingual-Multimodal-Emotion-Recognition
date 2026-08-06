from __future__ import annotations

import csv
import wave
from collections import Counter
from pathlib import Path

import pytest

from mmer.data import build_cremad_manifests, load_manifest, parse_cremad_basename
from mmer.data.cremad import CREMAD_TRANSCRIPTS, select_pilot_actors, stratified_actor_split


def _official_demographics() -> dict[str, str]:
    actors = [str(value) for value in range(1001, 1092)]
    return {actor: "Male" if index < 48 else "Female" for index, actor in enumerate(actors)}


def _split_counts(assignments: dict[str, str]) -> Counter[str]:
    return Counter(assignments.values())


def _sex_counts(assignments: dict[str, str], demographics: dict[str, str], split: str) -> Counter[str]:
    return Counter(demographics[actor] for actor, value in assignments.items() if value == split)


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(b"\x00\x00" * 160)


def _write_fake_flv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"FLV" + b"\x00" * 300)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_parse_cremad_basename():
    parsed = parse_cremad_basename("1001_DFA_ANG_XX.wav")
    assert parsed.actor_id == "1001"
    assert parsed.sentence_code == "DFA"
    assert parsed.source_emotion == "ANG"
    assert parsed.intensity == "XX"
    assert len(CREMAD_TRANSCRIPTS) == 12

    with pytest.raises(ValueError, match="invalid CREMA-D filename"):
        parse_cremad_basename("bad-name.wav")


def test_full_and_pilot_speaker_policy_is_exact_and_deterministic():
    demographics = _official_demographics()
    first = stratified_actor_split(demographics, seed=17)
    second = stratified_actor_split(demographics, seed=17)

    assert first == second
    assert _split_counts(first) == {"train": 64, "validation": 14, "test": 13}
    assert _sex_counts(first, demographics, "train") == {"Male": 34, "Female": 30}
    assert _sex_counts(first, demographics, "validation") == {"Male": 7, "Female": 7}
    assert _sex_counts(first, demographics, "test") == {"Male": 7, "Female": 6}

    pilot = select_pilot_actors(first, demographics, seed=17)
    assert Counter(first[actor] for actor in pilot) == {"train": 8, "validation": 2, "test": 2}
    for split in ("train", "validation", "test"):
        assert Counter(demographics[actor] for actor in pilot if first[actor] == split) == {
            "Male": (4 if split == "train" else 1),
            "Female": (4 if split == "train" else 1),
        }


def test_build_cremad_manifests_end_to_end(tmp_path: Path):
    dataset = tmp_path / "data" / "raw" / "crema_d"
    demographics = _official_demographics()
    demographic_rows = [
        {
            "ActorID": actor,
            "Age": "30",
            "Sex": sex,
            "Race": "Test",
            "Ethnicity": "Test",
        }
        for actor, sex in demographics.items()
    ]
    _write_csv(
        dataset / "VideoDemographics.csv",
        ["ActorID", "Age", "Sex", "Race", "Ethnicity"],
        demographic_rows,
    )

    filenames: list[str] = []
    sources = (("IEO", "ANG"), ("TIE", "HAP"), ("IOM", "NEU"), ("DFA", "SAD"))
    for actor in demographics:
        for sentence, emotion in sources:
            basename = f"{actor}_{sentence}_{emotion}_XX"
            filenames.append(basename)
            _write_wav(dataset / "AudioWAV" / f"{basename}.wav")
            _write_fake_flv(dataset / "VideoFlash" / f"{basename}.flv")
    excluded = "1001_TAI_DIS_XX"
    filenames.append(excluded)
    _write_wav(dataset / "AudioWAV" / f"{excluded}.wav")
    _write_fake_flv(dataset / "VideoFlash" / f"{excluded}.flv")
    _write_csv(
        dataset / "SentenceFilenames.csv",
        ["Stimulus_Number", "Filename"],
        [
            {"Stimulus_Number": str(index), "Filename": basename}
            for index, basename in enumerate(filenames, start=1)
        ],
    )

    result = build_cremad_manifests(tmp_path, check_files=True)
    full = load_manifest(result.full_manifest)
    pilot = load_manifest(result.pilot_manifest)

    assert len(full) == 91 * 4
    assert len(pilot) == 12 * 4
    assert Counter(sample.emotion for sample in full) == {
        "angry": 91,
        "happy": 91,
        "neutral": 91,
        "sad": 91,
    }
    assert Counter(sample.split for sample in full) == {
        "train": 64 * 4,
        "validation": 14 * 4,
        "test": 13 * 4,
    }
    assert all(sample.modality_mask() == (True, True, True) for sample in full)
    assert result.report["speaker_split_counts"]["full"] == {
        "train": 64,
        "validation": 14,
        "test": 13,
    }
    assert result.report["speaker_split_counts"]["pilot"] == {
        "train": 8,
        "validation": 2,
        "test": 2,
    }
    assert result.report["excluded"] == {"source_emotion:DIS": 1}
    assert not result.report["media_problems"]
