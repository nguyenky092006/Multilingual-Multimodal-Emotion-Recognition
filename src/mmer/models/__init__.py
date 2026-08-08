"""Projection, temporal pooling, and multimodal classifier modules."""

from .temporal import MaskedTemporalMean, TemporalAttentionPool
from .trimodal import TrimodalEmotionModel

__all__ = ["MaskedTemporalMean", "TemporalAttentionPool", "TrimodalEmotionModel"]
