"""Deterministic synthetic cached trimodal embeddings for offline tests."""

from __future__ import annotations

from typing import Mapping

import torch

from .cached import CachedEmbeddingDataset, CachedExample, MODALITIES

DEFAULT_LABELS = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}


def make_synthetic_dataset(
    size: int,
    input_dims: Mapping[str, int],
    seed: int,
    labels: Mapping[str, int] | None = None,
    split: str = "synthetic",
) -> CachedEmbeddingDataset:
    """Create learnable but non-scientific cached embeddings with missing modalities."""

    if size < 4:
        raise ValueError("synthetic dataset needs at least four samples")
    label_map = dict(labels or DEFAULT_LABELS)
    generator = torch.Generator().manual_seed(seed)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda pair: pair[1])]
    languages = ("en", "zh", "unknown")
    corpora = ("synthetic_a", "synthetic_b", "unknown")
    mask_patterns = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=torch.bool,
    )
    examples: list[CachedExample] = []
    for index in range(size):
        label_index = index % len(label_names)
        emotion = label_names[label_index]
        language = languages[index % len(languages)]
        corpus = corpora[(index // len(languages)) % len(corpora)]
        mask = mask_patterns[index % len(mask_patterns)].clone()
        quality = 0.25 + 0.75 * torch.rand(3, generator=generator)
        quality = quality * mask.float()
        embeddings: dict[str, torch.Tensor] = {}
        for modality_index, modality in enumerate(MODALITIES):
            dimension = int(input_dims[modality])
            vector = 0.20 * torch.randn(dimension, generator=generator)
            # A small deterministic class signal makes the smoke loop test learning,
            # while language/corpus offsets exercise the routed residuals.
            vector[label_index % dimension] += 1.5
            vector[(label_index + 4) % dimension] += 0.3 if language == "zh" else -0.3
            vector[(label_index + 7) % dimension] += 0.2 if corpus == "synthetic_b" else -0.2
            if not mask[modality_index]:
                vector.zero_()
            embeddings[modality] = vector
        examples.append(
            CachedExample(
                sample_id=f"synthetic-{seed}-{index:04d}",
                embeddings=embeddings,
                modality_mask=mask,
                quality=quality,
                label=label_index,
                emotion=emotion,
                language=language,
                corpus=corpus,
                speaker_id=f"{split}-speaker-{index // len(label_names):03d}",
                split=split,
                metadata={"transcript_source": "synthetic"},
            )
        )
    return CachedEmbeddingDataset(examples)
