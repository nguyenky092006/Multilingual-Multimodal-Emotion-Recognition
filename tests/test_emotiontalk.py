from __future__ import annotations

import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

from mmer.data import (
    build_emotiontalk_manifests,
    load_emotiontalk_records,
    load_manifest,
    speaker_components,
)


def _write_stereo_wav(path: Path, duration: float = 0.02) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 44_100
    frames = int(rate * duration)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x00\x00\x00\x00" * frames)


def _write_record(
    root: Path,
    source: str,
    dialogue: str,
    speaker: str,
    index: int,
    emotion: str,
) -> None:
    turn = f"{dialogue}_{speaker}"
    stem = f"{turn}_{index:03d}"
    relative = Path(source) / dialogue / turn / f"{stem}.wav"
    wav_path = root / "Audio" / "wav" / relative
    json_path = root / "Audio" / "json" / relative.with_suffix(".json")
    _write_stereo_wav(wav_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": {
            "A": {"emotion": emotion, "Confidence_degree": "8"},
            "B": {"emotion": emotion, "Confidence_degree": "7"},
            "C": {"emotion": emotion, "Confidence_degree": "9"},
        },
        "speaker_id": speaker,
        "emotion_result": emotion,
        "content": f"这是中文测试句子 {source} {speaker} {index}",
        "paragraphs": {"startTime": 0.0, "endTime": 0.02, "duration": 0.02},
        "sourceAttr": {},
        "file_path": relative.as_posix(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_fixture(project_root: Path) -> Path:
    dataset = project_root / "data" / "raw" / "emotiontalk"
    labels = ("angry", "happy", "neutral", "sad")
    components = (
        ("G00001", "G00001_01", ("01", "02")),
        ("G00002", "G00002_01", ("03", "04")),
        ("G00003", "G00003_01", ("05", "06")),
    )
    for source, dialogue, speakers in components:
        for speaker_index, speaker in enumerate(speakers):
            for label_index, label in enumerate(labels):
                _write_record(
                    dataset,
                    source,
                    dialogue,
                    speaker,
                    speaker_index * len(labels) + label_index + 1,
                    label,
                )
    _write_record(dataset, "G00001", "G00001_01", "01", 99, "fearful")
    return dataset


def test_record_loading_and_connected_components(tmp_path: Path):
    dataset = _make_fixture(tmp_path)
    records = load_emotiontalk_records(dataset)

    assert len(records) == 25
    assert speaker_components(records) == [("01", "02"), ("03", "04"), ("05", "06")]
    assert all(record.transcript.startswith("这是中文测试句子") for record in records)
    assert all(record.audio_path.suffix == ".wav" for record in records)


def test_build_emotiontalk_manifests_end_to_end(tmp_path: Path):
    _make_fixture(tmp_path)
    result = build_emotiontalk_manifests(
        tmp_path,
        pilot_samples_per_class={"train": 1, "validation": 1, "test": 1},
        check_files=True,
    )
    full = load_manifest(result.full_manifest)
    pilot = load_manifest(result.pilot_manifest)

    assert len(full) == 24
    assert len(pilot) == 12
    assert Counter(sample.split for sample in full) == {
        "train": 8,
        "validation": 8,
        "test": 8,
    }
    assert Counter(sample.emotion for sample in full) == {
        "angry": 6,
        "happy": 6,
        "neutral": 6,
        "sad": 6,
    }
    assert all(sample.language == "zh" and sample.corpus == "emotiontalk" for sample in full)
    assert all(sample.modality_mask() == (True, True, False) for sample in full)
    assert all(sample.transcript_source == "official_audio_transcript" for sample in full)

    speaker_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for sample in full:
        speaker_splits[sample.speaker_id].add(sample.split)
        source_splits[str(sample.source_video_id)].add(sample.split)
    assert all(len(values) == 1 for values in speaker_splits.values())
    assert all(len(values) == 1 for values in source_splits.values())
    assert result.report["excluded"] == {"source_emotion:fearful": 1}
    assert result.report["leakage_audit"] == {
        "speaker_overlap": False,
        "source_group_overlap": False,
        "components_kept_whole": True,
    }
    assert result.report["raw"]["wav_contracts"] == [
        {"channels": 2, "sample_width_bytes": 2, "sample_rate": 44_100, "files": 25}
    ]


def test_split_and_pilot_are_deterministic(tmp_path: Path):
    _make_fixture(tmp_path)
    first = build_emotiontalk_manifests(
        tmp_path,
        output_dir="data/manifests/first",
        pilot_samples_per_class={"train": 1, "validation": 1, "test": 1},
        check_files=False,
    )
    second = build_emotiontalk_manifests(
        tmp_path,
        output_dir="data/manifests/second",
        pilot_samples_per_class={"train": 1, "validation": 1, "test": 1},
        check_files=False,
    )

    first_full = [(sample.sample_id, sample.split) for sample in load_manifest(first.full_manifest)]
    second_full = [(sample.sample_id, sample.split) for sample in load_manifest(second.full_manifest)]
    first_pilot = [sample.sample_id for sample in load_manifest(first.pilot_manifest)]
    second_pilot = [sample.sample_id for sample in load_manifest(second.pilot_manifest)]
    assert first_full == second_full
    assert first_pilot == second_pilot
