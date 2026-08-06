"""UAR, macro-F1, accuracy, and confusion matrix."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def confusion_matrix(targets: Sequence[int] | Tensor, predictions: Sequence[int] | Tensor, num_classes: int) -> Tensor:
    true = torch.as_tensor(targets, dtype=torch.long).flatten()
    predicted = torch.as_tensor(predictions, dtype=torch.long).flatten()
    if true.shape != predicted.shape:
        raise ValueError("targets and predictions must have equal shape")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if true.numel() and ((true < 0).any() or (true >= num_classes).any() or (predicted < 0).any() or (predicted >= num_classes).any()):
        raise ValueError("class index is outside configured label range")
    indices = true * num_classes + predicted
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def classification_metrics(
    targets: Sequence[int] | Tensor,
    predictions: Sequence[int] | Tensor,
    num_classes: int,
) -> dict[str, object]:
    matrix = confusion_matrix(targets, predictions, num_classes).float()
    support = matrix.sum(dim=1)
    predicted_count = matrix.sum(dim=0)
    true_positive = matrix.diag()
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    precision = torch.where(predicted_count > 0, true_positive / predicted_count, torch.zeros_like(predicted_count))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    total = matrix.sum()
    accuracy = float(true_positive.sum() / total) if total > 0 else 0.0
    return {
        "uar": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "accuracy": accuracy,
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.to(torch.long).tolist(),
        "support": support.to(torch.long).tolist(),
    }

