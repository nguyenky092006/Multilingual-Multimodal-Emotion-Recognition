"""Episodic sampling, Prototypical Networks, and optional alignment losses."""

from .episodes import Episode, EpisodeSampler
from .losses import supervised_contrastive_loss
from .prototypical import (
    PrototypicalResult,
    class_prototypes,
    prototypical_logits,
    prototypical_loss,
)

__all__ = [
    "Episode",
    "EpisodeSampler",
    "PrototypicalResult",
    "class_prototypes",
    "prototypical_logits",
    "prototypical_loss",
    "supervised_contrastive_loss",
]
