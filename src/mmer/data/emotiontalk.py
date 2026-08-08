"""EmotionTalk audio/text conversion with leakage-safe component splits."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .manifest import ManifestSample, write_manifest
from .validation import validate_manifest

EMOTIONTALK_LABELS = {
    "angry": "angry",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class EmotionTalkRecord:
    sample_id: str
    source_id: str
    dialogue_id: str
    speaker_id: str
    raw_emotion: str
    transcript: str
    duration: float
    audio_path: Path
    annotator_emotions: tuple[str, ...]


@dataclass(slots=True)
class EmotionTalkBuildResult:
    full_manifest: Path
    pilot_manifest: Path
    split_file: Path
    report_file: Path
    report: dict[str, Any]


def _resolve_under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes the expected root: {value}") from exc
    return resolved


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"media path is outside project root: {path}") from exc


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_paths(root: Path) -> Iterable[Path]:
    for base, directories, filenames in os.walk(root):
        directories.sort()
        for filename in sorted(filenames):
            if filename.casefold().endswith(".json"):
                yield Path(base) / filename


def _read_record(json_path: Path, json_root: Path, wav_root: Path) -> EmotionTalkRecord:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise ValueError(f"invalid EmotionTalk JSON: {json_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"EmotionTalk JSON must contain an object: {json_path}")

    relative_json = json_path.relative_to(json_root)
    if len(relative_json.parts) < 4:
        raise ValueError(f"unexpected EmotionTalk JSON layout: {relative_json}")
    source_id, dialogue_id, speaker_turn_id = relative_json.parts[:3]
    speaker_id = str(payload.get("speaker_id", "")).strip()
    if not speaker_id:
        raise ValueError(f"missing speaker_id: {relative_json}")
    expected_speaker = speaker_turn_id.rsplit("_", 1)[-1]
    if speaker_id != expected_speaker:
        raise ValueError(
            f"speaker mismatch for {relative_json}: JSON={speaker_id!r}, path={expected_speaker!r}"
        )

    raw_emotion = str(payload.get("emotion_result", "")).strip().casefold()
    if not raw_emotion:
        raise ValueError(f"missing emotion_result: {relative_json}")
    transcript = str(payload.get("content", "")).strip()
    if not transcript:
        raise ValueError(f"empty official transcript: {relative_json}")
    paragraphs = payload.get("paragraphs")
    if not isinstance(paragraphs, dict):
        raise ValueError(f"missing paragraphs metadata: {relative_json}")
    try:
        duration = float(paragraphs["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid duration: {relative_json}") from exc
    if duration <= 0:
        raise ValueError(f"duration must be positive: {relative_json}")

    raw_file_path = str(payload.get("file_path", "")).replace("\\", "/").strip()
    file_path = PurePosixPath(raw_file_path)
    if not raw_file_path or file_path.is_absolute() or ".." in file_path.parts:
        raise ValueError(f"unsafe or missing file_path in {relative_json}: {raw_file_path!r}")
    expected_wav = relative_json.with_suffix(".wav").as_posix()
    if file_path.as_posix() != expected_wav:
        raise ValueError(
            f"audio/JSON path mismatch for {relative_json}: {file_path.as_posix()!r}"
        )
    audio_path = _resolve_under(wav_root, Path(*file_path.parts))
    if not audio_path.is_file() or audio_path.stat().st_size <= 44:
        raise FileNotFoundError(f"missing or empty EmotionTalk WAV: {audio_path}")

    annotations = payload.get("data", {})
    annotator_emotions: list[str] = []
    if isinstance(annotations, dict):
        for value in annotations.values():
            if isinstance(value, dict) and value.get("emotion"):
                annotator_emotions.append(str(value["emotion"]).strip().casefold())
    return EmotionTalkRecord(
        sample_id=f"emotiontalk-{json_path.stem}",
        source_id=source_id,
        dialogue_id=dialogue_id,
        speaker_id=speaker_id,
        raw_emotion=raw_emotion,
        transcript=transcript,
        duration=duration,
        audio_path=audio_path,
        annotator_emotions=tuple(annotator_emotions),
    )


def load_emotiontalk_records(dataset_root: str | Path) -> list[EmotionTalkRecord]:
    """Read and structurally validate all Audio JSON/WAV pairs."""

    source = Path(dataset_root).resolve()
    json_root = source / "Audio" / "json"
    wav_root = source / "Audio" / "wav"
    missing = [str(path) for path in (json_root, wav_root) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"EmotionTalk Audio extraction is incomplete; missing: {missing}")
    records = [_read_record(path, json_root, wav_root) for path in _json_paths(json_root)]
    if not records:
        raise ValueError("EmotionTalk Audio/json contains no JSON records")
    ids = Counter(record.sample_id for record in records)
    duplicates = sorted(key for key, count in ids.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate EmotionTalk sample IDs: {duplicates[:5]}")
    return records


def speaker_components(records: Sequence[EmotionTalkRecord]) -> list[tuple[str, ...]]:
    """Return connected speaker components induced by shared dialogues."""

    dialogue_speakers: dict[str, set[str]] = defaultdict(set)
    graph: dict[str, set[str]] = defaultdict(set)
    for record in records:
        dialogue_speakers[record.dialogue_id].add(record.speaker_id)
        graph[record.speaker_id]
    for speakers in dialogue_speakers.values():
        for speaker in speakers:
            graph[speaker].update(speakers - {speaker})
    seen: set[str] = set()
    components: list[tuple[str, ...]] = []
    for speaker in sorted(graph):
        if speaker in seen:
            continue
        pending = [speaker]
        seen.add(speaker)
        component: list[str] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbour in sorted(graph[current]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    pending.append(neighbour)
        components.append(tuple(sorted(component)))
    return sorted(components, key=lambda value: (-len(value), value))


def connected_component_split(
    records: Sequence[EmotionTalkRecord],
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
) -> tuple[dict[str, str], list[tuple[str, ...]]]:
    """Choose a deterministic sample-balanced assignment of whole speaker components."""

    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("three positive split ratios are required")
    ratio_total = sum(ratios)
    targets = tuple(value / ratio_total for value in ratios)
    components = speaker_components(records)
    if len(components) < 3:
        raise ValueError("at least three disconnected speaker components are required")
    filtered = [record for record in records if record.raw_emotion in EMOTIONTALK_LABELS]
    if not filtered:
        raise ValueError("no approved four-class EmotionTalk records were found")
    global_labels = Counter(record.raw_emotion for record in filtered)
    global_total = len(filtered)
    global_distribution = {
        label: global_labels[label] / global_total for label in EMOTIONTALK_LABELS
    }
    component_index = {
        speaker: index for index, component in enumerate(components) for speaker in component
    }
    component_samples = Counter(component_index[record.speaker_id] for record in filtered)
    component_labels: dict[int, Counter[str]] = defaultdict(Counter)
    for record in filtered:
        component_labels[component_index[record.speaker_id]][record.raw_emotion] += 1

    best: tuple[float, tuple[int, ...]] | None = None
    for assignment in itertools.product(range(3), repeat=len(components)):
        if set(assignment) != {0, 1, 2} or assignment[0] != 0:
            continue
        split_counts = [sum(component_samples[i] for i, split in enumerate(assignment) if split == j) for j in range(3)]
        if split_counts[0] < max(split_counts[1:]):
            continue
        if any(count == 0 for count in split_counts):
            continue
        split_labels = [Counter() for _ in range(3)]
        split_speakers = [0, 0, 0]
        for index, split in enumerate(assignment):
            split_labels[split].update(component_labels[index])
            split_speakers[split] += len(components[index])
        if any(any(counts[label] == 0 for label in EMOTIONTALK_LABELS) for counts in split_labels):
            continue
        sample_error = sum((split_counts[i] / global_total - targets[i]) ** 2 for i in range(3))
        speaker_total = sum(split_speakers)
        speaker_error = sum((split_speakers[i] / speaker_total - targets[i]) ** 2 for i in range(3))
        class_error = 0.0
        for index, counts in enumerate(split_labels):
            for label in EMOTIONTALK_LABELS:
                class_error += (counts[label] / split_counts[index] - global_distribution[label]) ** 2
        score = sample_error + 0.05 * class_error + 0.01 * speaker_error
        candidate = (round(score, 15), assignment)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no valid three-way component split contains every approved label")

    speaker_splits: dict[str, str] = {}
    for index, split_index in enumerate(best[1]):
        for speaker in components[index]:
            speaker_splits[speaker] = SPLITS[split_index]
    return speaker_splits, components


def select_pilot_samples(
    samples: Sequence[ManifestSample],
    per_class: Mapping[str, int],
    seed: int = 17,
) -> list[ManifestSample]:
    """Select a deterministic class-balanced subset within the leakage-safe full split."""

    selected: list[ManifestSample] = []
    for split in SPLITS:
        requested = int(per_class[split])
        if requested <= 0:
            raise ValueError(f"pilot count for {split} must be positive")
        for label in EMOTIONTALK_LABELS.values():
            candidates = [sample for sample in samples if sample.split == split and sample.emotion == label]
            if len(candidates) < requested:
                raise ValueError(
                    f"pilot requests {requested} {split}/{label} samples but only {len(candidates)} exist"
                )
            candidates.sort(
                key=lambda sample: hashlib.sha256(
                    f"{seed}:{sample.sample_id}".encode("utf-8")
                ).hexdigest()
            )
            selected.extend(candidates[:requested])
    return sorted(selected, key=lambda sample: sample.sample_id)


def _check_wav(record: EmotionTalkRecord) -> dict[str, Any]:
    try:
        with wave.open(str(record.audio_path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            rate = stream.getframerate()
            frames = stream.getnframes()
    except (EOFError, OSError, wave.Error) as exc:
        raise ValueError(f"corrupt WAV for {record.sample_id}: {exc}") from exc
    if channels <= 0 or sample_width <= 0 or rate <= 0 or frames <= 0:
        raise ValueError(f"invalid WAV header for {record.sample_id}")
    wav_duration = frames / rate
    if abs(wav_duration - record.duration) > 0.1:
        raise ValueError(
            f"duration mismatch for {record.sample_id}: JSON={record.duration}, WAV={wav_duration}"
        )
    return {"channels": channels, "sample_width_bytes": sample_width, "sample_rate": rate}


def _count_by(samples: Sequence[ManifestSample], attribute: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(sample, attribute)) for sample in samples).items()))


def _warning_counts(report: Any) -> dict[str, int]:
    return dict(sorted(Counter(issue.code for issue in report.issues if issue.severity != "error").items()))


def _annotation_audit(records: Sequence[EmotionTalkRecord]) -> dict[str, Any]:
    with_votes = 0
    result_in_majority = 0
    for record in records:
        counts = Counter(record.annotator_emotions)
        if not counts:
            continue
        with_votes += 1
        maximum = max(counts.values())
        if record.raw_emotion in {label for label, count in counts.items() if count == maximum}:
            result_in_majority += 1
    return {
        "records_with_annotator_votes": with_votes,
        "emotion_result_in_majority": result_in_majority,
        "rate": result_in_majority / with_votes if with_votes else None,
    }


def build_emotiontalk_manifests(
    project_root: str | Path,
    dataset_root: str | Path = "data/raw/emotiontalk",
    output_dir: str | Path = "data/manifests",
    seed: int = 17,
    split_ratios: Sequence[float] = (0.70, 0.15, 0.15),
    pilot_samples_per_class: Mapping[str, int] | None = None,
    check_files: bool = True,
) -> EmotionTalkBuildResult:
    """Build full and diagnostic pilot manifests from extracted EmotionTalk Audio data."""

    root = Path(project_root).resolve()
    source = Path(dataset_root)
    source = (root / source).resolve() if not source.is_absolute() else source.resolve()
    output = Path(output_dir)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    records = load_emotiontalk_records(source)
    speaker_splits, components = connected_component_split(records, split_ratios)

    source_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_splits[record.source_id].add(speaker_splits[record.speaker_id])
    leaking_sources = {source_id: values for source_id, values in source_splits.items() if len(values) > 1}
    if leaking_sources:
        raise ValueError(f"source groups span component splits: {leaking_sources}")

    wav_contracts: Counter[tuple[int, int, int]] = Counter()
    if check_files:
        for record in records:
            contract = _check_wav(record)
            wav_contracts[(contract["channels"], contract["sample_width_bytes"], contract["sample_rate"])] += 1

    exclusions: Counter[str] = Counter()
    samples: list[ManifestSample] = []
    for record in records:
        if record.raw_emotion not in EMOTIONTALK_LABELS:
            exclusions[f"source_emotion:{record.raw_emotion}"] += 1
            continue
        samples.append(
            ManifestSample(
                sample_id=record.sample_id,
                audio_path=_relative(record.audio_path, root),
                video_path=None,
                transcript=record.transcript,
                emotion=EMOTIONTALK_LABELS[record.raw_emotion],
                speaker_id=record.speaker_id,
                language="zh",
                corpus="emotiontalk",
                split=speaker_splits[record.speaker_id],
                duration=record.duration,
                transcript_source="official_audio_transcript",
                asr_confidence=None,
                audio_available=True,
                text_available=True,
                visual_available=False,
                country=None,
                region=None,
                accent=None,
                source_video_id=record.source_id,
                source_emotion=record.raw_emotion,
                sentence_code=None,
                intensity=None,
            )
        )
    samples.sort(key=lambda sample: sample.sample_id)
    if not samples:
        raise ValueError("no approved four-class EmotionTalk samples were found")
    pilot_counts = dict(pilot_samples_per_class or {"train": 64, "validation": 16, "test": 16})
    if set(pilot_counts) != set(SPLITS):
        raise ValueError(f"pilot_samples_per_class must define exactly {SPLITS}")
    pilot = select_pilot_samples(samples, pilot_counts, seed)

    output.mkdir(parents=True, exist_ok=True)
    full_path = output / "emotiontalk_full.jsonl"
    pilot_path = output / "emotiontalk_pilot.jsonl"
    split_path = output / "emotiontalk_speaker_splits.json"
    report_path = output / "emotiontalk_build_report.json"
    write_manifest(samples, full_path)
    write_manifest(pilot, pilot_path)

    labels = set(EMOTIONTALK_LABELS.values())
    full_validation = validate_manifest(samples, labels, root=root, check_files=False)
    pilot_validation = validate_manifest(pilot, labels, root=root, check_files=False)
    full_validation.raise_for_errors()
    pilot_validation.raise_for_errors()

    component_payload: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        split_names = {speaker_splits[speaker] for speaker in component}
        split = next(iter(split_names))
        component_samples = [sample for sample in samples if sample.speaker_id in component]
        component_sources = sorted(
            {record.source_id for record in records if record.speaker_id in component}
        )
        component_payload.append(
            {
                "component": index,
                "split": split,
                "speakers": list(component),
                "sources": component_sources,
                "four_class_samples": len(component_samples),
                "emotion_counts": _count_by(component_samples, "emotion"),
            }
        )
    split_payload = {
        "strategy": "connected_speaker_components_balanced_by_four_class_sample_count",
        "ratios": list(split_ratios),
        "speaker_to_split": dict(sorted(speaker_splits.items())),
        "components": component_payload,
    }
    split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    config_hash = hashlib.sha256(
        json.dumps(
            {
                "seed": seed,
                "split_ratios": list(split_ratios),
                "pilot_samples_per_class": pilot_counts,
                "labels": EMOTIONTALK_LABELS,
                "target_source": "Audio.emotion_result",
                "transcript_source": "Audio.content",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    split_speakers = {
        split: sorted(speaker for speaker, value in speaker_splits.items() if value == split)
        for split in SPLITS
    }
    split_sources = {
        split: sorted(source_id for source_id, values in source_splits.items() if values == {split})
        for split in SPLITS
    }
    report: dict[str, Any] = {
        "dataset": "EmotionTalk",
        "license": "CC BY-NC-SA 4.0 (verify the upstream dataset card before publication)",
        "seed": seed,
        "config_hash": config_hash,
        "target_policy": "Audio.emotion_result; keep angry/happy/neutral/sad; exclude without merging",
        "transcript_policy": "Audio.content in original Chinese; Text.tar labels are not used as targets",
        "raw": {
            "records": len(records),
            "emotion_counts": dict(sorted(Counter(record.raw_emotion for record in records).items())),
            "speakers": len({record.speaker_id for record in records}),
            "dialogues": len({record.dialogue_id for record in records}),
            "sources": len({record.source_id for record in records}),
            "annotation_audit": _annotation_audit(records),
            "wav_contracts": [
                {
                    "channels": key[0],
                    "sample_width_bytes": key[1],
                    "sample_rate": key[2],
                    "files": count,
                }
                for key, count in sorted(wav_contracts.items())
            ],
        },
        "full": {
            "samples": len(samples),
            "split_sample_counts": _count_by(samples, "split"),
            "emotion_counts": _count_by(samples, "emotion"),
            "speaker_split_counts": {split: len(split_speakers[split]) for split in SPLITS},
            "source_split_counts": {split: len(split_sources[split]) for split in SPLITS},
            "speakers": split_speakers,
            "sources": split_sources,
            "validation_warning_counts": _warning_counts(full_validation),
        },
        "pilot": {
            "samples": len(pilot),
            "samples_per_class_requested": pilot_counts,
            "split_sample_counts": _count_by(pilot, "split"),
            "emotion_counts": _count_by(pilot, "emotion"),
            "validation_warning_counts": _warning_counts(pilot_validation),
        },
        "excluded": dict(sorted(exclusions.items())),
        "leakage_audit": {
            "speaker_overlap": False,
            "source_group_overlap": False,
            "components_kept_whole": True,
        },
        "outputs": {
            "full_manifest": _display_path(full_path, root),
            "pilot_manifest": _display_path(pilot_path, root),
            "speaker_splits": _display_path(split_path, root),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return EmotionTalkBuildResult(full_path, pilot_path, split_path, report_path, report)
