"""Portable MMER checkpoint save/load using PyTorch's restricted unpickler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metadata: Mapping[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "metadata": dict(metadata),
    }
    torch.save(payload, output)
    return output


def read_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    required = {"model_state", "optimizer_state", "epoch", "metadata"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("invalid MMER checkpoint")
    if not isinstance(payload["model_state"], dict) or not isinstance(payload["metadata"], dict):
        raise ValueError("invalid MMER checkpoint payload types")
    return payload


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = read_checkpoint(path, map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload["optimizer_state"] is not None:
        if not isinstance(payload["optimizer_state"], dict):
            raise ValueError("invalid optimizer checkpoint state")
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
