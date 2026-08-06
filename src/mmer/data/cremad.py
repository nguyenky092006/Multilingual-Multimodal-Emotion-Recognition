"""CREMA-D to MMER JSONL conversion with speaker-exclusive splits."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import ManifestSample, write_manifest
from .validation import validate_manifest

CREMAD_PATTERN = re.compile(
    r"^(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>[A-Z]{3})_(?P<intensity>[A-Z]{2})$"
)

CREMAD_LABELS = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

CREMAD_TRANSCRIPTS = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class CremadName:
    actor_id: str
    sentence_code: str
    source_emotion: str
    intensity: str


@dataclass(slots=True)
class CremadBuildResult:
    full_manifest: Path
    pilot_manifest: Path
    split_file: Path
    report_file: Path
    report: dict[str, Any]


def parse_cremad_basename(value: str) -> CremadName:
    """Parse a CREMA-D filename without its extension."""

    match = CREMAD_PATTERN.fullmatch(Path(value).stem.upper())
    if match is None:
        raise ValueError(f"invalid CREMA-D filename: {value!r}")
    return CremadName(
        actor_id=match.group("actor"),
        sentence_code=match.group("sentence"),
        source_emotion=match.group("emotion"),
        intensity=match.group("intensity"),
    )


def _largest_remainder_counts(total: int, ratios: Sequence[float]) -> list[int]:
    if total < 0 or not ratios or any(value < 0 for value in ratios):
        raise ValueError("invalid split count/ratios")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("split ratios must have a positive sum")
    normalised = [value / ratio_sum for value in ratios]
    raw = [total * value for value in normalised]
    result = [int(value) for value in raw]
    remaining = total - sum(result)
    order = sorted(range(len(raw)), key=lambda index: (raw[index] - result[index], -index), reverse=True)
    for index in order[:remaining]:
        result[index] += 1
    return result


def stratified_actor_split(
    actor_to_sex: Mapping[str, str],
    seed: int = 17,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
) -> dict[str, str]:
    """Assign every actor to one split while approximately preserving sex strata."""

    if len(ratios) != len(SPLITS):
        raise ValueError("three split ratios are required")
    strata: dict[str, list[str]] = defaultdict(list)
    for actor, sex in actor_to_sex.items():
        if not actor:
            raise ValueError("actor ID cannot be empty")
        strata[(sex or "unknown").casefold()].append(str(actor))
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for sex in sorted(strata):
        actors = sorted(strata[sex])
        rng.shuffle(actors)
        counts = _largest_remainder_counts(len(actors), ratios)
        offset = 0
        for split, count in zip(SPLITS, counts, strict=True):
            for actor in actors[offset : offset + count]:
                assignments[actor] = split
            offset += count
    if set(assignments) != set(actor_to_sex):
        raise RuntimeError("not every CREMA-D actor received a split")
    return assignments


def select_pilot_actors(
    actor_splits: Mapping[str, str],
    actor_to_sex: Mapping[str, str],
    counts: Mapping[str, int] | None = None,
    seed: int = 17,
) -> set[str]:
    """Select a deterministic, sex-balanced actor subset within the full split."""

    requested = dict(counts or {"train": 8, "validation": 2, "test": 2})
    selected: set[str] = set()
    rng = random.Random(seed + 1_000)
    for split in SPLITS:
        target = int(requested[split])
        candidates = [actor for actor, value in actor_splits.items() if value == split]
        if target > len(candidates):
            raise ValueError(f"pilot requests {target} {split} actors but only {len(candidates)} exist")
        by_sex: dict[str, list[str]] = defaultdict(list)
        for actor in candidates:
            by_sex[(actor_to_sex.get(actor) or "unknown").casefold()].append(actor)
        for values in by_sex.values():
            values.sort()
            rng.shuffle(values)
        sex_names = sorted(by_sex, key=lambda key: len(by_sex[key]), reverse=True)
        split_selected: list[str] = []
        while len(split_selected) < target:
            made_progress = False
            for sex in sex_names:
                if by_sex[sex] and len(split_selected) < target:
                    split_selected.append(by_sex[sex].pop())
                    made_progress = True
            if not made_progress:
                break
        if len(split_selected) != target:
            raise RuntimeError(f"could not select balanced pilot actors for {split}")
        selected.update(split_selected)
    return selected


def _read_demographics(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    actor_to_sex: dict[str, str] = {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            actor = str(row.get("ActorID", "")).strip()
            if not actor:
                raise ValueError("VideoDemographics.csv contains an empty ActorID")
            actor_to_sex[actor] = str(row.get("Sex", "unknown")).strip() or "unknown"
            rows[actor] = {str(key): str(value) for key, value in row.items() if key is not None}
    if not rows:
        raise ValueError("VideoDemographics.csv contains no actors")
    return actor_to_sex, rows


def _read_official_filenames(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get("Filename", "")).strip()
            if value:
                values.add(value.upper())
    if not values:
        raise ValueError("SentenceFilenames.csv contains no filenames")
    return values


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        frames = stream.getnframes()
    if rate <= 0 or frames <= 0:
        raise ValueError(f"invalid WAV duration: {path}")
    return frames / rate


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"media path is outside project root: {path}") from exc


def _count_by(samples: Sequence[ManifestSample], attribute: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(sample, attribute)) for sample in samples).items()))


def _speaker_audit(
    actor_splits: Mapping[str, str], actor_to_sex: Mapping[str, str], pilot_actors: set[str]
) -> dict[str, Any]:
    payload: dict[str, Any] = {"full": {}, "pilot": {}}
    for split in SPLITS:
        full = sorted(actor for actor, value in actor_splits.items() if value == split)
        pilot = sorted(actor for actor in full if actor in pilot_actors)
        payload["full"][split] = {
            "actors": full,
            "sex_counts": dict(sorted(Counter(actor_to_sex[actor] for actor in full).items())),
        }
        payload["pilot"][split] = {
            "actors": pilot,
            "sex_counts": dict(sorted(Counter(actor_to_sex[actor] for actor in pilot).items())),
        }
    return payload


def build_cremad_manifests(
    project_root: str | Path,
    dataset_root: str | Path = "data/raw/crema_d",
    output_dir: str | Path = "data/manifests",
    seed: int = 17,
    split_ratios: Sequence[float] = (0.70, 0.15, 0.15),
    pilot_speaker_counts: Mapping[str, int] | None = None,
    check_files: bool = True,
) -> CremadBuildResult:
    """Build full and pilot four-class manifests from an official CREMA-D clone."""

    root = Path(project_root).resolve()
    source = Path(dataset_root)
    source = (root / source).resolve() if not source.is_absolute() else source.resolve()
    output = Path(output_dir)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    required = [
        source / "AudioWAV",
        source / "VideoFlash",
        source / "SentenceFilenames.csv",
        source / "VideoDemographics.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"CREMA-D is incomplete; missing: {missing}")

    actor_to_sex, demographics = _read_demographics(source / "VideoDemographics.csv")
    official_filenames = _read_official_filenames(source / "SentenceFilenames.csv")
    actor_splits = stratified_actor_split(actor_to_sex, seed, split_ratios)
    pilot_actors = select_pilot_actors(actor_splits, actor_to_sex, pilot_speaker_counts, seed)

    exclusions: Counter[str] = Counter()
    media_problems: list[dict[str, str]] = []
    samples: list[ManifestSample] = []
    for basename in sorted(official_filenames):
        parsed = parse_cremad_basename(basename)
        if parsed.source_emotion not in CREMAD_LABELS:
            exclusions[f"source_emotion:{parsed.source_emotion}"] += 1
            continue
        if parsed.actor_id not in actor_splits:
            raise ValueError(f"actor {parsed.actor_id} is absent from VideoDemographics.csv")
        if parsed.sentence_code not in CREMAD_TRANSCRIPTS:
            raise ValueError(f"unknown CREMA-D sentence code: {parsed.sentence_code}")

        audio_path = source / "AudioWAV" / f"{basename}.wav"
        video_path = source / "VideoFlash" / f"{basename}.flv"
        audio_available = audio_path.is_file() and audio_path.stat().st_size > 256
        visual_available = video_path.is_file() and video_path.stat().st_size > 256
        duration: float | None = None
        if audio_available:
            try:
                duration = _wav_duration(audio_path)
            except (EOFError, OSError, ValueError, wave.Error) as exc:
                audio_available = False
                media_problems.append({"sample": basename, "modality": "audio", "problem": str(exc)})
        else:
            media_problems.append(
                {"sample": basename, "modality": "audio", "problem": "missing or LFS-pointer audio"}
            )
        if not visual_available:
            media_problems.append(
                {"sample": basename, "modality": "visual", "problem": "missing or LFS-pointer video"}
            )
        samples.append(
            ManifestSample(
                sample_id=f"cremad-{basename}",
                audio_path=_relative(audio_path, root) if audio_available else None,
                video_path=_relative(video_path, root) if visual_available else None,
                transcript=CREMAD_TRANSCRIPTS[parsed.sentence_code],
                emotion=CREMAD_LABELS[parsed.source_emotion],
                speaker_id=parsed.actor_id,
                language="en",
                corpus="crema-d",
                split=actor_splits[parsed.actor_id],
                duration=duration,
                transcript_source="official_prompt",
                asr_confidence=None,
                audio_available=audio_available,
                text_available=True,
                visual_available=visual_available,
                country=None,
                region=None,
                accent=None,
                source_video_id=basename,
                source_emotion=parsed.source_emotion,
                sentence_code=parsed.sentence_code,
                intensity=parsed.intensity,
            )
        )

    if not samples:
        raise ValueError("no four-class CREMA-D samples were found")
    pilot = [sample for sample in samples if sample.speaker_id in pilot_actors]
    output.mkdir(parents=True, exist_ok=True)
    full_path = output / "cremad_full.jsonl"
    pilot_path = output / "cremad_pilot.jsonl"
    split_path = output / "cremad_speaker_splits.json"
    report_path = output / "cremad_build_report.json"
    write_manifest(samples, full_path)
    write_manifest(pilot, pilot_path)

    full_report = validate_manifest(samples, set(CREMAD_LABELS.values()), root=root, check_files=check_files)
    pilot_report = validate_manifest(pilot, set(CREMAD_LABELS.values()), root=root, check_files=check_files)
    full_report.raise_for_errors()
    pilot_report.raise_for_errors()
    speaker_payload = _speaker_audit(actor_splits, actor_to_sex, pilot_actors)
    split_path.write_text(json.dumps(speaker_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    config_hash = hashlib.sha256(
        json.dumps(
            {
                "seed": seed,
                "split_ratios": list(split_ratios),
                "pilot_speaker_counts": dict(pilot_speaker_counts or {"train": 8, "validation": 2, "test": 2}),
                "labels": CREMAD_LABELS,
                "transcripts": CREMAD_TRANSCRIPTS,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    report: dict[str, Any] = {
        "dataset": "CREMA-D",
        "label_policy": "intended four-class labels; DIS and FEA excluded without merging",
        "seed": seed,
        "config_hash": config_hash,
        "full": {
            "samples": len(samples),
            "split_sample_counts": _count_by(samples, "split"),
            "emotion_counts": _count_by(samples, "emotion"),
            "intensity_counts": _count_by(samples, "intensity"),
            "modality_patterns": dict(sorted(Counter("".join("ATV"[i] for i, flag in enumerate(s.modality_mask()) if flag) for s in samples).items())),
            "validation_warnings": [asdict(issue) for issue in full_report.issues if issue.severity != "error"],
        },
        "pilot": {
            "samples": len(pilot),
            "split_sample_counts": _count_by(pilot, "split"),
            "emotion_counts": _count_by(pilot, "emotion"),
            "validation_warnings": [asdict(issue) for issue in pilot_report.issues if issue.severity != "error"],
        },
        "speaker_split_counts": {
            "full": {split: len(speaker_payload["full"][split]["actors"]) for split in SPLITS},
            "pilot": {split: len(speaker_payload["pilot"][split]["actors"]) for split in SPLITS},
        },
        "excluded": dict(sorted(exclusions.items())),
        "media_problems": media_problems,
        "demographic_fields_audited_not_modelled": sorted(next(iter(demographics.values())).keys()),
        "outputs": {
            "full_manifest": _relative(full_path, root),
            "pilot_manifest": _relative(pilot_path, root),
            "speaker_splits": _relative(split_path, root),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return CremadBuildResult(full_path, pilot_path, split_path, report_path, report)
