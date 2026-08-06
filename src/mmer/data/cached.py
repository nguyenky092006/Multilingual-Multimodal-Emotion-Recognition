"""In-memory and file-backed cached embedding datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

MODALITIES = ("audio", "text", "visual")


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


def save_cache(examples: Sequence[CachedExample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(examples), output)


def load_cache(path: str | Path) -> CachedEmbeddingDataset:
    examples = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(examples, list) or not all(isinstance(item, CachedExample) for item in examples):
        raise ValueError("cache does not contain CachedExample records")
    return CachedEmbeddingDataset(examples)


def apply_modality_dropout(mask: Tensor, probability: float, generator: torch.Generator | None = None) -> Tensor:
    """Randomly drop available modalities while preserving at least one per sample."""

    if not 0.0 <= probability < 1.0:
        raise ValueError("modality dropout probability must be in [0, 1)")
    output = mask.bool().clone()
    if probability == 0.0:
        return output
    random_values = torch.rand(output.shape, generator=generator)
    random_values = random_values.to(output.device)
    output &= random_values >= probability
    all_dropped = ~output.any(dim=1)
    for row_index in all_dropped.nonzero(as_tuple=False).flatten().tolist():
        candidates = mask[row_index].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("input modality mask contains an all-missing sample")
        chosen_index = torch.randint(candidates.numel(), (1,), generator=generator).item()
        chosen = candidates[chosen_index]
        output[row_index, chosen] = True
    return output
