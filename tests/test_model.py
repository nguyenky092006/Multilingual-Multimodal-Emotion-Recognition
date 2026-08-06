from __future__ import annotations

import pytest
import torch

from mmer.models import TrimodalEmotionModel
from mmer.models.trimodal import parameter_counts

DIMS = {"audio": 12, "text": 10, "visual": 8}


def model(fusion: str = "reliability") -> TrimodalEmotionModel:
    return TrimodalEmotionModel(DIMS, 4, ["en", "zh"], ["a", "b"], 16, 24, 4, 0.0, fusion)


def inputs(batch: int = 4):
    return {name: torch.randn(batch, dimension) for name, dimension in DIMS.items()}


def test_classifier_output_dimensions_and_routes():
    network = model()
    mask = torch.ones(4, 3, dtype=torch.bool)
    output = network(inputs(), mask, ["en", "zh", "fr", "en"], ["a", "b", "new", "a"], torch.rand(4, 3))
    assert output["logits"].shape == (4, 4)
    assert output["fusion_weights"].shape == (4, 3)
    assert set(output["route_stats"]) == {"audio", "text", "visual"}


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_missing_modality_inference(missing_index):
    network = model()
    mask = torch.ones(2, 3, dtype=torch.bool)
    mask[:, missing_index] = False
    output = network(inputs(2), mask, ["en", "zh"], ["a", "b"], torch.rand(2, 3))
    assert torch.isfinite(output["logits"]).all()
    assert torch.equal(output["fusion_weights"][:, missing_index], torch.zeros(2))


def test_parameter_count_is_reported():
    counts = parameter_counts(model())
    assert counts["total"] == counts["trainable"]
    assert counts["trainable_percent"] == 100.0


def test_parameter_matched_separate_emotion_adapters():
    network = TrimodalEmotionModel(DIMS, 4, ["en"], ["a"], 16, 24, 4, 0.0, "concat", False)
    output = network(inputs(1), torch.ones(1, 3, dtype=torch.bool), ["en"], ["a"], torch.ones(1, 3))
    assert output["logits"].shape == (1, 4)

