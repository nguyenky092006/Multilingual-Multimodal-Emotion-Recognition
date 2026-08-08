"""Manifest integrity, leakage, and data-quality checks."""

from __future__ import annotations

import re
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from .manifest import ManifestSample


SERIOUS_CODES = {
    "duplicate_sample_id",
    "duplicate_audio_across_splits",
    "duplicate_video_across_splits",
    "speaker_split_overlap",
    "speaker_train_test_overlap",
    "source_video_split_overlap",
    "no_available_modality",
    "unsupported_label",
    "missing_availability_metadata",
    "missing_required_value",
    "availability_path_mismatch",
    "availability_text_mismatch",
    "invalid_asr_confidence",
    "invalid_duration",
    "invalid_split",
    "path_outside_root",
    "missing_audio",
    "missing_video",
    "empty_audio",
    "empty_video",
    "corrupt_audio",
    "corrupt_video",
}


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "warning"
    sample_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    sample_count: int = 0
    language_corpus_counts: dict[str, int] = field(default_factory=dict)

    @property
    def serious(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def is_valid(self) -> bool:
        return not self.serious

    def raise_for_errors(self) -> None:
        if self.serious:
            summary = "; ".join(f"{item.code}: {item.message}" for item in self.serious)
            raise ManifestValidationError(summary)


class ManifestValidationError(RuntimeError):
    """Raised when serious leakage or schema problems must stop training."""


def _normalise_transcript(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold(), flags=re.UNICODE)


def _group_values(
    samples: Iterable[ManifestSample], attribute: str
) -> dict[str, list[ManifestSample]]:
    grouped: dict[str, list[ManifestSample]] = defaultdict(list)
    for sample in samples:
        value = getattr(sample, attribute)
        if value:
            grouped[str(value)].append(sample)
    return grouped


def _check_media(path: Path, kind: str) -> str | None:
    if not path.exists() or not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    if kind == "audio" and path.suffix.casefold() == ".wav":
        try:
            with wave.open(str(path), "rb") as stream:
                if stream.getnframes() <= 0 or stream.getframerate() <= 0:
                    return "corrupt"
        except (wave.Error, EOFError, OSError):
            return "corrupt"
    if kind == "video" and path.suffix.casefold() not in {
        ".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm",
    }:
        return "unsupported_container"
    if kind == "video":
        try:
            with path.open("rb") as handle:
                header = handle.read(16)
        except OSError:
            return "corrupt"
        suffix = path.suffix.casefold()
        valid_header = {
            ".mp4": len(header) >= 8 and header[4:8] == b"ftyp",
            ".mov": len(header) >= 8 and header[4:8] == b"ftyp",
            ".avi": header.startswith(b"RIFF") and header[8:12] == b"AVI ",
            ".flv": header.startswith(b"FLV"),
            ".mkv": header.startswith(b"\x1a\x45\xdf\xa3"),
            ".webm": header.startswith(b"\x1a\x45\xdf\xa3"),
        }.get(suffix, False)
        if not valid_header:
            return "corrupt"
    return None


def validate_manifest(
    samples: Iterable[ManifestSample],
    labels: set[str],
    root: str | Path | None = None,
    check_files: bool = True,
    near_duplicate_threshold: float = 0.94,
) -> ValidationReport:
    """Validate schema, all-split leakage, paths, and non-fatal dataset risks."""

    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    items = list(samples)
    report = ValidationReport(sample_count=len(items))

    def add(code: str, message: str, ids: Iterable[str] = ()) -> None:
        report.issues.append(
            ValidationIssue(code, message, "error" if code in SERIOUS_CODES else "warning", tuple(ids))
        )

    for sample_id, group in _group_values(items, "sample_id").items():
        if len(group) > 1:
            add("duplicate_sample_id", f"sample_id {sample_id!r} appears {len(group)} times", [sample_id])

    for item in items:
        for field_name in ("sample_id", "speaker_id", "language", "corpus", "split"):
            value = getattr(item, field_name)
            if not isinstance(value, str) or not value.strip():
                add(
                    "missing_required_value",
                    f"{item.sample_id}: {field_name} must be a non-empty string",
                    [item.sample_id],
                )
        if item.split not in {"train", "validation", "test"}:
            add("invalid_split", f"{item.sample_id}: unsupported split {item.split!r}", [item.sample_id])
        if item.emotion not in labels:
            add("unsupported_label", f"{item.sample_id}: unsupported emotion {item.emotion!r}", [item.sample_id])
        mask = item.modality_mask()
        if not all(isinstance(value, bool) for value in mask):
            add("missing_availability_metadata", f"{item.sample_id}: availability fields must be booleans", [item.sample_id])
        if not any(mask):
            add("no_available_modality", f"{item.sample_id}: all modalities are unavailable", [item.sample_id])
        if item.audio_available and not item.audio_path:
            add("availability_path_mismatch", f"{item.sample_id}: audio available but audio_path is null", [item.sample_id])
        if item.visual_available and not item.video_path:
            add("availability_path_mismatch", f"{item.sample_id}: visual available but video_path is null", [item.sample_id])
        if item.text_available and not (item.transcript and item.transcript.strip()):
            add("availability_text_mismatch", f"{item.sample_id}: text available but transcript is empty", [item.sample_id])
        if item.asr_confidence is not None and not 0.0 <= item.asr_confidence <= 1.0:
            add("invalid_asr_confidence", f"{item.sample_id}: asr_confidence must be in [0, 1]", [item.sample_id])
        if item.duration is not None and item.duration <= 0:
            add("invalid_duration", f"{item.sample_id}: duration must be positive", [item.sample_id])

    for attribute, code in (
        ("audio_path", "duplicate_audio_across_splits"),
        ("video_path", "duplicate_video_across_splits"),
    ):
        for path_value, group in _group_values(items, attribute).items():
            splits = {sample.split for sample in group}
            if len(splits) > 1:
                add(code, f"{path_value!r} occurs in splits {sorted(splits)}", [sample.sample_id for sample in group])
            elif len(group) > 1:
                within_code = "duplicate_audio" if attribute == "audio_path" else "duplicate_video"
                add(within_code, f"{path_value!r} is reused by {len(group)} samples", [sample.sample_id for sample in group])

    for speaker, group in _group_values(items, "speaker_id").items():
        splits = {sample.split.casefold() for sample in group}
        if len(splits) > 1:
            add("speaker_split_overlap", f"speaker {speaker!r} appears in {sorted(splits)}", [sample.sample_id for sample in group])
        if "train" in splits and "test" in splits:
            add("speaker_train_test_overlap", f"speaker {speaker!r} appears in train and test", [sample.sample_id for sample in group])

    for source_id, group in _group_values(items, "source_video_id").items():
        splits = {sample.split for sample in group}
        if len(splits) > 1:
            add("source_video_split_overlap", f"source video {source_id!r} spans {sorted(splits)}", [sample.sample_id for sample in group])

    transcript_groups: dict[str, list[ManifestSample]] = defaultdict(list)
    for item in items:
        if item.transcript:
            transcript_groups[_normalise_transcript(item.transcript)].append(item)
    for text_key, group in transcript_groups.items():
        if text_key and len(group) > 1 and len({sample.split for sample in group}) > 1:
            add("duplicate_transcript_across_splits", f"exact transcript duplicate spans splits: {text_key[:40]!r}", [sample.sample_id for sample in group])

    unique_texts = [(key, group[0]) for key, group in transcript_groups.items() if key]
    if len(unique_texts) <= 2_000:
        for index, (left, left_sample) in enumerate(unique_texts):
            for right, right_sample in unique_texts[index + 1 :]:
                if left_sample.split == right_sample.split or min(len(left), len(right)) < 8:
                    continue
                ratio = SequenceMatcher(None, left, right).ratio()
                if near_duplicate_threshold <= ratio < 1.0:
                    add("near_duplicate_transcript", f"{left_sample.sample_id}/{right_sample.sample_id}: similarity={ratio:.3f}", [left_sample.sample_id, right_sample.sample_id])

    label_counts = Counter(sample.emotion for sample in items)
    if label_counts and min(label_counts.values()) > 0:
        ratio = max(label_counts.values()) / min(label_counts.values())
        if ratio >= 3.0:
            add("class_imbalance", f"maximum/minimum class count ratio is {ratio:.2f}: {dict(label_counts)}")

    contingency = Counter(f"{sample.language}|{sample.corpus}" for sample in items)
    report.language_corpus_counts = dict(sorted(contingency.items()))
    language_to_corpora: dict[str, set[str]] = defaultdict(set)
    for item in items:
        language_to_corpora[item.language].add(item.corpus)
    if len(language_to_corpora) > 1 and all(len(values) == 1 for values in language_to_corpora.values()):
        add("language_corpus_confounding", f"every language occurs in one corpus: {dict(language_to_corpora)}")

    if check_files:
        base = Path(root).resolve() if root is not None else Path.cwd().resolve()
        for item in items:
            for available, raw_path, kind in (
                (item.audio_available, item.audio_path, "audio"),
                (item.visual_available, item.video_path, "video"),
            ):
                if not available or not raw_path:
                    continue
                resolved = (base / raw_path).resolve()
                try:
                    resolved.relative_to(base)
                except ValueError:
                    add("path_outside_root", f"{item.sample_id}: {kind} path escapes project root: {raw_path}", [item.sample_id])
                    continue
                problem = _check_media(resolved, kind)
                if problem:
                    add(f"{problem}_{kind}", f"{item.sample_id}: {kind} file {problem}: {raw_path}", [item.sample_id])
    return report
