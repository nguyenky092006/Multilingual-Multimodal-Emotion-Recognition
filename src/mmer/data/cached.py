"""In-memory and safely serialized cached embedding datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

MODALITIES = ("audio", "text", "visual")
CACHE_SCHEMA_VERSION = 4


@dataclass(slots=True)
class CachedExample:
    sample_id: str
    embeddings: dict[str, Tensor]
    modality_mask: Tensor
    quality: Tensor
    label: int
    emotion: str
    language: str
    corpus: str
    speaker_id: str = "unknown"
    split: str = "unknown"
    metadata: dict[str, str | None] = field(default_factory=dict)


class CachedEmbeddingDataset(Dataset[CachedExample]):
    """A dataset whose encoder representations are already cached."""

    def __init__(self, examples: Sequence[CachedExample]) -> None:
        if not examples:
            raise ValueError("cached dataset cannot be empty")
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CachedExample:
        return self.examples[index]


def _collate_embedding(values: Sequence[Tensor], modality: str) -> tuple[Tensor, Tensor | None]:
    dimensions = {value.ndim for value in values}
    if dimensions == {1}:
        return torch.stack(list(values)), None
    if dimensions != {2}:
        raise ValueError(f"{modality} cache mixes vector and temporal representations")
    feature_dims = {int(value.shape[1]) for value in values}
    if len(feature_dims) != 1 or any(value.shape[0] <= 0 for value in values):
        raise ValueError(f"{modality} temporal embeddings have inconsistent dimensions")
    max_steps = max(int(value.shape[0]) for value in values)
    features = next(iter(feature_dims))
    batch = torch.zeros(len(values), max_steps, features, dtype=torch.float32)
    mask = torch.zeros(len(values), max_steps, dtype=torch.bool)
    for index, value in enumerate(values):
        steps = int(value.shape[0])
        batch[index, :steps] = value.float()
        mask[index, :steps] = True
    return batch, mask


def collate_cached(examples: Sequence[CachedExample]) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    embeddings: dict[str, Tensor] = {}
    temporal_masks: dict[str, Tensor] = {}
    for modality in MODALITIES:
        batch, temporal_mask = _collate_embedding(
            [item.embeddings[modality] for item in examples], modality
        )
        embeddings[modality] = batch
        if temporal_mask is not None:
            temporal_masks[modality] = temporal_mask
    return {
        "sample_ids": [item.sample_id for item in examples],
        "embeddings": embeddings,
        "temporal_masks": temporal_masks,
        "modality_mask": torch.stack([item.modality_mask for item in examples]).bool(),
        "quality": torch.stack([item.quality for item in examples]).float(),
        "labels": torch.tensor([item.label for item in examples], dtype=torch.long),
        "emotions": [item.emotion for item in examples],
        "languages": [item.language for item in examples],
        "corpora": [item.corpus for item in examples],
        "speaker_ids": [item.speaker_id for item in examples],
        "splits": [item.split for item in examples],
        "metadata": [dict(item.metadata) for item in examples],
    }


def _record(item: CachedExample) -> dict[str, Any]:
    return {
        "sample_id": item.sample_id,
        "embeddings": {name: item.embeddings[name].detach().cpu() for name in MODALITIES},
        "modality_mask": item.modality_mask.detach().cpu().bool(),
        "quality": item.quality.detach().cpu().float(),
        "label": int(item.label),
        "emotion": item.emotion,
        "language": item.language,
        "corpus": item.corpus,
        "speaker_id": item.speaker_id,
        "split": item.split,
        "metadata": dict(item.metadata),
    }


def save_cache(examples: Sequence[CachedExample], path: str | Path) -> None:
    """Save primitive records so loading can use PyTorch's restricted unpickler."""

    if not examples:
        raise ValueError("cannot save an empty cache")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"schema_version": CACHE_SCHEMA_VERSION, "examples": [_record(item) for item in examples]},
        output,
    )


def _example(record: Any, schema_version: int) -> CachedExample:
    if not isinstance(record, dict):
        raise ValueError("cache example must be a mapping")
    legacy_required = {
        "sample_id", "embeddings", "modality_mask", "quality", "label",
        "emotion", "language", "corpus",
    }
    required = legacy_required | {"speaker_id", "split"}
    current_required = required | {"metadata"}
    if schema_version == 1:
        expected = legacy_required
    elif schema_version <= 3:
        expected = required
    else:
        expected = current_required
    if set(record) != expected:
        raise ValueError("cache example fields do not match the schema")
    embeddings = record["embeddings"]
    if not isinstance(embeddings, dict) or set(embeddings) != set(MODALITIES):
        raise ValueError("cache example must contain audio/text/visual embeddings")
    tensors = {name: embeddings[name] for name in MODALITIES}
    allowed_dimensions = {1} if schema_version <= 2 else {1, 2}
    if not all(
        isinstance(value, Tensor)
        and value.ndim in allowed_dimensions
        and value.numel() > 0
        for value in tensors.values()
    ):
        raise ValueError("cached embeddings must be non-empty vectors or temporal matrices")
    mask = record["modality_mask"]
    quality = record["quality"]
    if not isinstance(mask, Tensor) or mask.shape != (len(MODALITIES),):
        raise ValueError("cached modality_mask must have shape [3]")
    if not isinstance(quality, Tensor) or quality.shape != (len(MODALITIES),):
        raise ValueError("cached quality must have shape [3]")
    if not all(torch.isfinite(value).all() for value in [*tensors.values(), quality]):
        raise ValueError("cache contains non-finite tensors")
    raw_metadata = record.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("cached metadata must be a mapping")
    return CachedExample(
        sample_id=str(record["sample_id"]),
        embeddings={name: value.float().contiguous() for name, value in tensors.items()},
        modality_mask=mask.bool(),
        quality=quality.float(),
        label=int(record["label"]),
        emotion=str(record["emotion"]),
        language=str(record["language"]),
        corpus=str(record["corpus"]),
        speaker_id=str(record.get("speaker_id", "unknown")),
        split=str(record.get("split", "unknown")),
        metadata={
            str(key): (None if value is None else str(value))
            for key, value in raw_metadata.items()
        },
    )


def load_cache(path: str | Path) -> CachedEmbeddingDataset:
    """Load only tensor/basic-type payloads; arbitrary pickle execution is disabled."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        1,
        2,
        3,
        CACHE_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported or invalid MMER cache schema")
    records = payload.get("examples")
    if not isinstance(records, list) or not records:
        raise ValueError("cache contains no examples")
    version = int(payload["schema_version"])
    return CachedEmbeddingDataset([_example(record, version) for record in records])


def apply_modality_dropout(
    mask: Tensor,
    probability: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly drop available modalities while preserving at least one per sample."""

    if mask.ndim != 2 or mask.shape[1] != len(MODALITIES):
        raise ValueError("input modality mask must have shape [batch, 3]")
    if not 0.0 <= probability < 1.0:
        raise ValueError("modality dropout probability must be in [0, 1)")
    output = mask.bool().clone()
    if (~output).all(dim=1).any():
        raise ValueError("input modality mask contains an all-missing sample")
    if probability == 0.0:
        return output
    random_values = torch.rand(output.shape, generator=generator).to(output.device)
    output &= random_values >= probability
    all_dropped = ~output.any(dim=1)
    for row_index in all_dropped.nonzero(as_tuple=False).flatten().tolist():
        candidates = mask[row_index].nonzero(as_tuple=False).flatten()
        chosen_index = torch.randint(candidates.numel(), (1,), generator=generator).item()
        output[row_index, candidates[chosen_index]] = True
    return output


def apply_modality_subset(mask: Tensor, modalities: Sequence[str] | None) -> Tensor:
    """Restrict a batch to an explicit modality subset for controlled stress tests."""

    if mask.ndim != 2 or mask.shape[1] != len(MODALITIES):
        raise ValueError("input modality mask must have shape [batch, 3]")
    if modalities is None:
        return mask.bool()
    requested = tuple(str(value) for value in modalities)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("modality subset must be non-empty and unique")
    unknown = set(requested) - set(MODALITIES)
    if unknown:
        raise ValueError(f"unsupported modality subset: {sorted(unknown)}")
    allowed = torch.tensor(
        [name in requested for name in MODALITIES], dtype=torch.bool, device=mask.device
    )
    output = mask.bool() & allowed.unsqueeze(0)
    if (~output).all(dim=1).any():
        raise ValueError("modality subset leaves at least one sample with no available input")
    return output
