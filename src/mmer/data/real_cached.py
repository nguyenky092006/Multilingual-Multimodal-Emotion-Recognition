"""Strict assembly of real multimodal SafeTensor caches into training datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

from .cached import MODALITIES, CachedEmbeddingDataset, CachedExample
from .manifest import ManifestSample, load_manifest


@dataclass(slots=True)
class RealCacheBundle:
    """Speaker-exclusive dataset splits plus their reproducibility audit."""

    splits: dict[str, CachedEmbeddingDataset]
    audit: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inside_root(root: Path, value: str | Path, description: str) -> Path:
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} must be inside project root: {path}") from exc
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON metadata root must be an object: {path}")
    return payload


def _load_index(path: Path, root: Path, modality: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {modality} cache index {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {modality} index line {line_number}: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            raise ValueError(f"invalid {modality} index record at line {line_number}")
        sample_id = row["sample_id"]
        if sample_id in rows:
            raise ValueError(f"duplicate sample_id in {modality} index: {sample_id}")
        cache_value = row.get("cache_path")
        if not isinstance(cache_value, str):
            raise ValueError(f"{modality} index record has no cache_path: {sample_id}")
        cache_path = _inside_root(root, cache_value, f"{modality} cache path")
        if not cache_path.is_file():
            raise ValueError(f"{modality} cache file is missing: {cache_path}")
        copied = dict(row)
        copied["_resolved_cache_path"] = cache_path
        rows[sample_id] = copied
    if not rows:
        raise ValueError(f"{modality} cache index is empty: {path}")
    return rows


def _contract_dimension(contract: Mapping[str, Any]) -> int | None:
    value = contract.get("embedding_dimension", contract.get("expected_embedding_dimension"))
    return None if value is None else int(value)


def _validate_cache_source(
    root: Path,
    modality: str,
    specification: Mapping[str, Any],
    manifest_hash: str,
    expected_dimension: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    for key in ("index_path", "contract_path"):
        if not isinstance(specification.get(key), str):
            raise ValueError(f"{modality} cache specification requires {key}")
    index_path = _inside_root(root, specification["index_path"], f"{modality} index")
    contract_path = _inside_root(root, specification["contract_path"], f"{modality} contract")
    contract = _load_json(contract_path)
    if contract.get("modality") != modality:
        raise ValueError(f"cache contract modality mismatch for {modality}: {contract_path}")
    if str(contract.get("cache_dtype")) != "float32":
        raise ValueError(f"{modality} cache must use float32 tensors")
    declared_dimension = _contract_dimension(contract)
    if declared_dimension is not None and declared_dimension != expected_dimension:
        raise ValueError(
            f"{modality} dimension mismatch: contract={declared_dimension}, "
            f"config={expected_dimension}"
        )
    contract_manifest_hash = contract.get("manifest_sha256")
    if contract_manifest_hash is None:
        summary_value = specification.get("summary_path")
        if not isinstance(summary_value, str):
            raise ValueError(
                f"{modality} contract is manifest-independent; a summary_path is required"
            )
        summary = _load_json(_inside_root(root, summary_value, f"{modality} summary"))
        contract_manifest_hash = summary.get("manifest_sha256")
    if contract_manifest_hash != manifest_hash:
        raise ValueError(
            f"{modality} cache was not built for the configured manifest: "
            f"{contract_manifest_hash} != {manifest_hash}"
        )
    rows = _load_index(index_path, root, modality)
    index_dimensions = {int(row.get("embedding_dimension", -1)) for row in rows.values()}
    if index_dimensions != {expected_dimension}:
        raise ValueError(
            f"{modality} index dimensions mismatch: {sorted(index_dimensions)} != "
            f"[{expected_dimension}]"
        )
    return rows, contract, _json_sha256(contract)


def _check_speaker_exclusivity(samples: Sequence[ManifestSample]) -> None:
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        speaker_splits[sample.speaker_id].add(sample.split)
    overlap = {
        speaker: sorted(splits)
        for speaker, splits in speaker_splits.items()
        if len(splits) > 1
    }
    if overlap:
        preview = dict(list(sorted(overlap.items()))[:5])
        raise ValueError(f"speaker leakage across manifest splits: {preview}")


def _metadata_quality(
    sample: ManifestSample,
    modality: str,
    row: Mapping[str, Any] | None,
    visual_frames_per_clip: int,
) -> float:
    if row is None:
        return 0.0
    if modality == "audio":
        clipping = float(row.get("clipping_fraction", 0.0))
        value = 1.0 - clipping
    elif modality == "text":
        if sample.asr_confidence is not None:
            value = float(sample.asr_confidence)
        elif (sample.transcript_source or "").lower() in {
            "gold", "manual", "provided", "scripted", "official_prompt",
        }:
            value = 1.0
        else:
            value = 0.0
    else:
        valid_frames = int(row.get("valid_frame_count", 0))
        value = valid_frames / visual_frames_per_clip
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"non-finite {modality} quality for {sample.sample_id}")
    return min(1.0, max(0.0, float(value)))


def _load_embedding(
    path: Path,
    expected_dimension: int,
    expected_contract_hash: str,
    memo: dict[Path, torch.Tensor],
) -> torch.Tensor:
    if path in memo:
        return memo[path]
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            vector = handle.get_tensor("embedding").clone()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot load cache tensor {path}: {exc}") from exc
    if metadata.get("contract_sha256") != expected_contract_hash:
        raise ValueError(f"cache tensor contract mismatch: {path}")
    if vector.shape != (expected_dimension,):
        raise ValueError(f"cache tensor dimension mismatch at {path}: {tuple(vector.shape)}")
    vector = vector.float().contiguous()
    if not torch.isfinite(vector).all():
        raise ValueError(f"cache tensor contains non-finite values: {path}")
    memo[path] = vector
    return vector


def load_real_cache_bundle(
    project_root: str | Path,
    manifest_path: str | Path,
    cache_sources: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, int],
    input_dims: Mapping[str, int],
    enabled_modalities: Sequence[str] = MODALITIES,
    quality_policy: str = "validated_metadata_v1",
) -> RealCacheBundle:
    """Join a manifest with cache indexes for enabled modalities only."""

    root = Path(project_root).resolve()
    manifest = _inside_root(root, manifest_path, "manifest")
    manifest_hash = _file_sha256(manifest)
    samples = load_manifest(manifest)
    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest contains duplicate sample_id values")
    _check_speaker_exclusivity(samples)

    requested = tuple(str(value) for value in enabled_modalities)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(MODALITIES):
        raise ValueError(f"invalid enabled_modalities: {requested}")
    enabled = tuple(name for name in MODALITIES if name in requested)
    unknown_dimensions = set(input_dims) - set(MODALITIES)
    if unknown_dimensions:
        raise ValueError(f"unsupported input dimensions: {sorted(unknown_dimensions)}")
    missing_dimensions = set(enabled) - set(input_dims)
    if missing_dimensions:
        raise ValueError(f"missing enabled input dimensions: {sorted(missing_dimensions)}")
    if any(int(input_dims[name]) <= 0 for name in input_dims):
        raise ValueError("input dimensions must be positive")
    dimensions = {name: int(input_dims.get(name, 1)) for name in MODALITIES}
    unknown_sources = set(cache_sources) - set(MODALITIES)
    if unknown_sources:
        raise ValueError(f"unsupported cache sources: {sorted(unknown_sources)}")
    missing_sources = set(enabled) - set(cache_sources)
    if missing_sources:
        raise ValueError(f"missing cache sources for enabled modalities: {sorted(missing_sources)}")
    if quality_policy != "validated_metadata_v1":
        raise ValueError(f"unsupported quality policy: {quality_policy}")

    index_rows: dict[str, dict[str, dict[str, Any]]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    contract_hashes: dict[str, str] = {}
    for modality in enabled:
        rows, contract, contract_hash = _validate_cache_source(
            root, modality, cache_sources[modality], manifest_hash, dimensions[modality]
        )
        available_ids = {
            sample.sample_id for sample in samples if getattr(sample, f"{modality}_available")
        }
        indexed_ids = set(rows)
        if indexed_ids != available_ids:
            missing = sorted(available_ids - indexed_ids)[:5]
            extra = sorted(indexed_ids - available_ids)[:5]
            raise ValueError(
                f"{modality} index/manifest sample mismatch; missing={missing}, extra={extra}"
            )
        index_rows[modality] = rows
        contracts[modality] = contract
        contract_hashes[modality] = contract_hash

    visual_frames = 1
    if "visual" in enabled:
        visual_frames = int(contracts["visual"].get("frames_per_clip", 0))
        if visual_frames <= 0:
            raise ValueError("visual cache contract has invalid frames_per_clip")

    memo: dict[Path, torch.Tensor] = {}
    split_examples: dict[str, list[CachedExample]] = defaultdict(list)
    qualities: dict[str, list[float]] = {name: [] for name in enabled}
    active_availability = Counter()
    manifest_availability = Counter()
    language_counts = Counter()
    corpus_counts = Counter()
    transcript_hashes: set[str] = set()
    speaker_counts: dict[str, set[str]] = defaultdict(set)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        language_counts[sample.language] += 1
        corpus_counts[sample.corpus] += 1
        speaker_counts[sample.split].add(sample.speaker_id)
        label_counts[sample.split][sample.emotion] += 1
        if sample.transcript:
            transcript_hashes.add(hashlib.sha256(sample.transcript.encode("utf-8")).hexdigest())
        mask_values: list[bool] = []
        quality_values: list[float] = []
        embeddings: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            available = bool(getattr(sample, f"{modality}_available"))
            if available:
                manifest_availability[modality] += 1
            active = available and modality in enabled
            if active:
                row = index_rows[modality].get(sample.sample_id)
                if row is None:
                    raise ValueError(f"missing {modality} cache row: {sample.sample_id}")
                vector = _load_embedding(
                    row["_resolved_cache_path"],
                    dimensions[modality],
                    contract_hashes[modality],
                    memo,
                )
                quality = _metadata_quality(sample, modality, row, visual_frames)
                active_availability[modality] += 1
            else:
                vector = torch.zeros(dimensions[modality], dtype=torch.float32)
                quality = 0.0
            embeddings[modality] = vector
            mask_values.append(active)
            quality_values.append(quality)
            if modality in enabled:
                qualities[modality].append(quality)
        if not any(mask_values):
            raise ValueError(f"sample has no enabled available modality: {sample.sample_id}")
        if sample.emotion not in labels:
            raise ValueError(f"unsupported emotion in manifest: {sample.emotion}")
        split_examples[sample.split].append(
            CachedExample(
                sample_id=sample.sample_id,
                embeddings=embeddings,
                modality_mask=torch.tensor(mask_values, dtype=torch.bool),
                quality=torch.tensor(quality_values, dtype=torch.float32),
                label=int(labels[sample.emotion]),
                emotion=sample.emotion,
                language=sample.language,
                corpus=sample.corpus,
            )
        )
    required_splits = {"train", "validation", "test"}
    if set(split_examples) != required_splits:
        raise ValueError(
            f"real cached training requires exactly {sorted(required_splits)}, "
            f"got {sorted(split_examples)}"
        )

    quality_summary = {
        modality: {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
        for modality, values in qualities.items()
    }
    audit = {
        "manifest": manifest.relative_to(root).as_posix(),
        "manifest_sha256": manifest_hash,
        "sample_count": len(samples),
        "split_counts": {key: len(value) for key, value in sorted(split_examples.items())},
        "speaker_counts": {key: len(value) for key, value in sorted(speaker_counts.items())},
        "label_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(label_counts.items())
        },
        "language_counts": dict(sorted(language_counts.items())),
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "unique_transcript_count": len(transcript_hashes),
        "enabled_modalities": list(enabled),
        "availability_counts": dict(active_availability),
        "manifest_availability_counts": dict(manifest_availability),
        "input_dims": dimensions,
        "quality_policy": quality_policy,
        "quality_summary": quality_summary,
        "cache_contracts": {
            modality: {
                "identifier": contracts[modality].get("model_identifier"),
                "resolved_revision": contracts[modality].get("resolved_revision"),
                "contract_sha256": contract_hashes[modality],
            }
            for modality in enabled
        },
        "unique_cache_tensors_loaded": len(memo),
    }
    return RealCacheBundle(
        splits={key: CachedEmbeddingDataset(value) for key, value in split_examples.items()},
        audit=audit,
    )
