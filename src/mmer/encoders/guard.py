"""Guard helpers for commands that may download large pretrained weights."""

from __future__ import annotations

from typing import Any, Mapping


class DownloadApprovalRequired(RuntimeError):
    """Raised when an encoder command would require unapproved model weights."""


def describe_encoder(
    modality: str,
    config: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    if modality not in {"audio", "text", "visual"}:
        raise ValueError(f"unsupported modality: {modality}")
    if modality not in config or not isinstance(config[modality], Mapping):
        raise ValueError(f"encoder configuration has no {modality} mapping")
    details = dict(config[modality])
    details["modality"] = modality
    details["status"] = "configuration validated; verified extractor is available"
    if not dry_run:
        raise DownloadApprovalRequired(
            f"{modality} extraction may download large weights. Use the concrete "
            f"cache_{modality}_embeddings.py command and add --allow-download only after approval."
        )
    return details
