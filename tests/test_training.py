from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mmer.data.cached import collate_cached
from mmer.data.synthetic import make_synthetic_dataset
from mmer.engine import evaluate_model, train_one_epoch
from mmer.metrics import classification_metrics
from mmer.models import TrimodalEmotionModel
from mmer.utils import load_checkpoint, save_checkpoint

DIMS = {"audio": 12, "text": 10, "visual": 8}


def make_model():
    return TrimodalEmotionModel(DIMS, 4, ["en", "zh"], ["synthetic_a", "synthetic_b"], 16, 24, 4, 0.0)


def test_uar_macro_f1_and_accuracy():
    metrics = classification_metrics([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 1, 2, 0, 3, 3], 4)
    assert metrics["uar"] == 0.75
    assert 0.0 < metrics["macro_f1"] <= 1.0
    assert metrics["accuracy"] == 0.75


def test_checkpoint_save_and_load(tmp_path):
    original = make_model()
    optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path / "checkpoint.pt", original, optimizer, 2, {"synthetic": True})
    restored = make_model()
    payload = load_checkpoint(path, restored)
    assert payload["epoch"] == 2
    for left, right in zip(original.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)


def test_one_cpu_training_and_evaluation_step():
    dataset = make_synthetic_dataset(16, DIMS, seed=9)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_cached)
    network = make_model()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    loss = train_one_epoch(network, loader, optimizer, torch.device("cpu"), modality_dropout=0.2)
    assert torch.isfinite(torch.tensor(loss))
    metrics = evaluate_model(network, loader, torch.device("cpu"), 4)
    assert set(("uar", "macro_f1", "accuracy", "fusion_weight_summary")) <= set(metrics)
    assert metrics["unavailable_weight_max"] == 0.0


def test_gradient_accumulation_and_cpu_bfloat16_training():
    dataset = make_synthetic_dataset(16, DIMS, seed=29)
    for example in dataset.examples:
        example.modality_mask[:] = True
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_cached)
    network = make_model()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    loss = train_one_epoch(
        network,
        loader,
        optimizer,
        torch.device("cpu"),
        gradient_accumulation_steps=2,
        mixed_precision="bfloat16",
    )
    assert torch.isfinite(torch.tensor(loss))
    metrics = evaluate_model(
        network,
        loader,
        torch.device("cpu"),
        4,
        modality_subset=["audio", "text"],
        mixed_precision="bfloat16",
    )
    assert metrics["forced_modality_subset"] == ["audio", "text"]
    assert metrics["unavailable_weight_max"] == 0.0
