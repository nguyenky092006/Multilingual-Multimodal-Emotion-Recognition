"""Frozen encoder specifications and verified cache extraction."""

from .audio import AudioCacheInput, AudioCacheResult, cache_audio_embeddings
from .guard import DownloadApprovalRequired, describe_encoder

__all__ = [
    "AudioCacheInput",
    "AudioCacheResult",
    "DownloadApprovalRequired",
    "cache_audio_embeddings",
    "describe_encoder",
]
