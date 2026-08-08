"""Prototype construction and differentiable episodic classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class PrototypicalResult:
    loss: Tensor
    logits: Tensor
    local_targets: Tensor
    predictions: Tensor
    prototypes: Tensor
    classes: Tensor


def class_prototypes(features: Tensor, labels: Tensor, classes: Tensor) -> Tensor:
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have shapes [samples, dim] and [samples]")
    rows: list[Tensor] = []
    for label in classes.tolist():
        selected = features[labels == int(label)]
        if selected.numel() == 0:
            raise ValueError(f"support set has no example for class {label}")
        rows.append(selected.mean(dim=0))
    return torch.stack(rows)


def prototypical_logits(
    query_features: Tensor,
    prototypes: Tensor,
    distance: str = "cosine",
    temperature: float = 0.1,
) -> Tensor:
    if query_features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("query features and prototypes must be matrices")
    if query_features.shape[1] != prototypes.shape[1] or temperature <= 0:
        raise ValueError("prototype dimensions must match and temperature must be positive")
    if distance == "cosine":
        similarities = F.normalize(query_features, dim=-1) @ F.normalize(prototypes, dim=-1).T
        return similarities / temperature
    if distance == "squared_euclidean":
        squared = (query_features[:, None, :] - prototypes[None, :, :]).square().sum(dim=-1)
        return -squared / temperature
    raise ValueError(f"unsupported prototypical distance: {distance}")


def prototypical_loss(
    support_features: Tensor,
    support_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
    classes: Tensor | None = None,
    distance: str = "cosine",
    temperature: float = 0.1,
) -> PrototypicalResult:
    episode_classes = torch.unique(support_labels, sorted=True) if classes is None else classes
    if episode_classes.ndim != 1 or episode_classes.numel() < 2:
        raise ValueError("prototypical episodes require at least two classes")
    prototypes = class_prototypes(support_features, support_labels, episode_classes)
    logits = prototypical_logits(query_features, prototypes, distance, temperature)
    mapping = {int(label): index for index, label in enumerate(episode_classes.tolist())}
    try:
        local_targets = torch.tensor(
            [mapping[int(label)] for label in query_labels.tolist()],
            dtype=torch.long,
            device=query_labels.device,
        )
    except KeyError as exc:
        raise ValueError("query set contains a class absent from support") from exc
    loss = nn.CrossEntropyLoss()(logits, local_targets)
    predictions = episode_classes.index_select(0, logits.argmax(dim=-1))
    return PrototypicalResult(loss, logits, local_targets, predictions, prototypes, episode_classes)
