"""Frozen encoder specifications and verified cache extraction."""

from .audio import AudioCacheInput, AudioCacheResult, cache_audio_embeddings
from .guard import DownloadApprovalRequired, describe_encoder
from .text import TextCacheInput, TextCacheResult, cache_text_embeddings, prompt_label_audit

__all__ = [
    "AudioCacheInput",
    "AudioCacheResult",
    "DownloadApprovalRequired",
    "TextCacheInput",
    "TextCacheResult",
    "cache_audio_embeddings",
    "cache_text_embeddings",
    "describe_encoder",
    "prompt_label_audit",
]
