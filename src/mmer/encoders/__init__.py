"""Frozen encoder specifications and verified cache extraction."""

from .audio import (
    AudioCacheInput,
    AudioCacheResult,
    attentive_statistics,
    cache_audio_embeddings,
    masked_mean,
    masked_statistics,
)
from .guard import DownloadApprovalRequired, describe_encoder
from .text import (
    TextCacheInput,
    TextCacheResult,
    cache_text_embeddings,
    masked_mean_pool,
    prompt_label_audit,
)
from .visual import (
    VisualCacheInput,
    VisualCacheResult,
    VisualClip,
    cache_visual_embeddings,
    decode_uniform_frames,
    opencv_haar_face_crop,
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
    "attentive_statistics",
    "cache_audio_embeddings",
    "cache_text_embeddings",
    "cache_visual_embeddings",
    "decode_uniform_frames",
    "describe_encoder",
    "masked_mean",
    "masked_mean_pool",
    "masked_statistics",
    "opencv_haar_face_crop",
    "prompt_label_audit",
    "uniform_frame_indices",
]
