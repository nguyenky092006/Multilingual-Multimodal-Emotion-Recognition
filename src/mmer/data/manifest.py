"""Typed JSONL manifest contract for independent MMER clips."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class ManifestSample:
    """One independent utterance/clip and its modality availability."""

    sample_id: str
    audio_path: str | None
    video_path: str | None
    transcript: str | None
    emotion: str
    speaker_id: str
    language: str
    corpus: str
    split: str
    duration: float | None
    transcript_source: str | None
    asr_confidence: float | None
    audio_available: bool
    text_available: bool
    visual_available: bool
    country: str | None = None
    region: str | None = None
    accent: str | None = None
    source_video_id: str | None = None

    def modality_mask(self) -> tuple[bool, bool, bool]:
        """Return availability in canonical audio/text/visual order."""

        return self.audio_available, self.text_available, self.visual_available

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManifestSample":
        names = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - names)
        if unknown:
            raise ValueError(f"unknown manifest fields: {unknown}")
        required = {
            "sample_id", "audio_path", "video_path", "transcript", "emotion",
            "speaker_id", "language", "corpus", "split", "duration",
            "transcript_source", "asr_confidence", "audio_available",
            "text_available", "visual_available",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing manifest fields: {missing}")
        return cls(**payload)


def load_manifest(path: str | Path) -> list[ManifestSample]:
    """Load a UTF-8 JSONL manifest, reporting the failing line."""

    manifest_path = Path(path)
    samples: list[ManifestSample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ValueError("each JSONL record must be an object")
                samples.append(ManifestSample.from_dict(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
    if not samples:
        raise ValueError("manifest contains no samples")
    return samples


def write_manifest(samples: Iterable[ManifestSample], path: str | Path) -> None:
    """Write a deterministic UTF-8 JSONL manifest."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
