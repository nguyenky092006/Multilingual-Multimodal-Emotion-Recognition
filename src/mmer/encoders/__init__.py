"""Frozen encoder specifications and verified cache extraction."""

from .audio import AudioCacheInput, AudioCacheResult, cache_audio_embeddings
from .guard import DownloadApprovalRequired, describe_encoder
from .text import TextCacheInput, TextCacheResult, cache_text_embeddings, prompt_label_audit
from .visual import (
    VisualCacheInput,
    VisualCacheResult,
    VisualClip,
    cache_visual_embeddings,
    decode_uniform_frames,
    uniform_frame_indices,
)

__all__ = [
    "AudioCacheInput",
    "AudioCacheResult",
    "DownloadApprovalRequired",
    "TextCacheInput",
    "TextCacheResult",
    "VisualCacheInput",
    "VisualCacheResult",
    "VisualClip",
    "cache_audio_embeddings",
    "cache_text_embeddings",
    "cache_visual_embeddings",
    "decode_uniform_frames",
    "describe_encoder",
    "prompt_label_audit",
    "uniform_frame_indices",
]
