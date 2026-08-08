"""Manifest, cached embedding, real-cache, and synthetic data utilities."""

from .cremad import build_cremad_manifests, parse_cremad_basename
from .emotiontalk import (
    build_emotiontalk_manifests,
    connected_component_split,
    load_emotiontalk_records,
    speaker_components,
)
from .manifest import ManifestSample, load_manifest, write_manifest
from .real_cached import RealCacheBundle, load_real_cache_bundle
from .validation import ManifestValidationError, ValidationReport, validate_manifest

__all__ = [
    "ManifestSample",
    "ManifestValidationError",
    "RealCacheBundle",
    "ValidationReport",
    "build_cremad_manifests",
    "build_emotiontalk_manifests",
    "connected_component_split",
    "load_manifest",
    "load_emotiontalk_records",
    "load_real_cache_bundle",
    "parse_cremad_basename",
    "speaker_components",
    "validate_manifest",
    "write_manifest",
]
