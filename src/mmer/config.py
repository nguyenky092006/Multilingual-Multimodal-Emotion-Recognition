"""YAML loading with explicit path and value checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {source}")
    return payload


def load_label_mapping(path: str | Path) -> dict[str, int]:
    payload = load_yaml(path)
    labels = payload.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("label YAML needs a non-empty labels mapping")
    mapping = {str(name): int(index) for name, index in labels.items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("label indices must be contiguous from zero")
    return mapping

