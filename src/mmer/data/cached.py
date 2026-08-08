"""In-memory and safely serialized cached embedding datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

MODALITIES = ("audio", "text", "visual")
CACHE_SCHEMA_VERSION = 1


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


def collate_cached(examples: Sequence[CachedExample]) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    return {
        "sample_ids": [item.sample_id for item in examples],
        "embeddings": {
            modality: torch.stack([item.embeddings[modality] for item in examples])
            for modality in MODALITIES
        },
        "modality_mask": torch.stack([item.modality_mask for item in examples]).bool(),
        "quality": torch.stack([item.quality for item in examples]).float(),
        "labels": torch.tensor([item.label for item in examples], dtype=torch.long),
        "emotions": [item.emotion for item in examples],
        "languages": [item.language for item in examples],
        "corpora": [item.corpus for item in examples],
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


def _example(record: Any) -> CachedExample:
    if not isinstance(record, dict):
        raise ValueError("cache example must be a mapping")
    required = {
        "sample_id", "embeddings", "modality_mask", "quality", "label",
        "emotion", "language", "corpus",
    }
    if set(record) != required:
        raise ValueError("cache example fields do not match the schema")
    embeddings = record["embeddings"]
    if not isinstance(embeddings, dict) or set(embeddings) != set(MODALITIES):
        raise ValueError("cache example must contain audio/text/visual embeddings")
    tensors = {name: embeddings[name] for name in MODALITIES}
    if not all(isinstance(value, Tensor) and value.ndim == 1 for value in tensors.values()):
        raise ValueError("cached embeddings must be one-dimensional tensors")
    mask = record["modality_mask"]
    quality = record["quality"]
    if not isinstance(mask, Tensor) or mask.shape != (len(MODALITIES),):
        raise ValueError("cached modality_mask must have shape [3]")
    if not isinstance(quality, Tensor) or quality.shape != (len(MODALITIES),):
        raise ValueError("cached quality must have shape [3]")
    if not all(torch.isfinite(value).all() for value in [*tensors.values(), quality]):
        raise ValueError("cache contains non-finite tensors")
    return CachedExample(
        sample_id=str(record["sample_id"]),
        embeddings={name: value.float().contiguous() for name, value in tensors.items()},
        modality_mask=mask.bool(),
        quality=quality.float(),
        label=int(record["label"]),
        emotion=str(record["emotion"]),
        language=str(record["language"]),
        corpus=str(record["corpus"]),
    )


def load_cache(path: str | Path) -> CachedEmbeddingDataset:
    """Load only tensor/basic-type payloads; arbitrary pickle execution is disabled."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported or invalid MMER cache schema")
    records = payload.get("examples")
    if not isinstance(records, list) or not records:
        raise ValueError("cache contains no examples")
    return CachedEmbeddingDataset([_example(record) for record in records])


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
