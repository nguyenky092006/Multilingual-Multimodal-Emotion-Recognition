from __future__ import annotations

import torch

from mmer.data.cached import apply_modality_dropout, load_cache, save_cache
from mmer.data.synthetic import make_synthetic_dataset

DIMS = {"audio": 12, "text": 10, "visual": 8}


def test_synthetic_data_contains_all_three_modalities():
    dataset = make_synthetic_dataset(28, DIMS, seed=3)
    masks = torch.stack([item.modality_mask for item in dataset.examples])
    assert masks.shape == (28, 3)
    assert masks.any(dim=0).all()


def test_synthetic_data_contains_each_missing_modality():
    dataset = make_synthetic_dataset(28, DIMS, seed=3)
    masks = torch.stack([item.modality_mask for item in dataset.examples])
    assert (~masks).any(dim=0).all()


def test_modality_dropout_never_drops_all():
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 1, 1]], dtype=torch.bool)
    generator = torch.Generator().manual_seed(2)
    for _ in range(20):
        output = apply_modality_dropout(mask, 0.95, generator)
        assert output.any(dim=1).all()
        assert not (output & ~mask).any()


def test_cached_dataset_round_trip(tmp_path):
    original = make_synthetic_dataset(8, DIMS, seed=4)
    path = tmp_path / "cache.pt"
    save_cache(original.examples, path)
    loaded = load_cache(path)
    assert len(loaded) == len(original)
    assert torch.equal(loaded[0].embeddings["visual"], original[0].embeddings["visual"])
