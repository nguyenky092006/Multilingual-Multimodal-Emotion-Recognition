"""Classification metrics without a scikit-learn runtime dependency."""

from .classification import classification_metrics, confusion_matrix

__all__ = ["classification_metrics", "confusion_matrix"]
