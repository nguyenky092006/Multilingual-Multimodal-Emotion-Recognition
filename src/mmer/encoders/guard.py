"""Prevent accidental large-model downloads in iteration one."""

from __future__ import annotations

from typing import Any, Mapping


class DownloadApprovalRequired(RuntimeError):
    """Raised when a cache command would require unapproved model weights."""


def describe_encoder(modality: str, config: Mapping[str, Any], dry_run: bool) -> dict[str, Any]:
    if modality not in {"audio", "text", "visual"}:
        raise ValueError(f"unsupported modality: {modality}")
    details = dict(config[modality])
    details["modality"] = modality
    details["status"] = "configuration validated; extraction intentionally deferred"
    if not dry_run:
        raise DownloadApprovalRequired(
            f"{modality} extraction can download large weights. Obtain explicit approval, "
            "then implement the verified dataset-specific extractor in iteration 2."
        )
    return details

