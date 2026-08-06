from __future__ import annotations

import pytest
import torch

from mmer.fusion import ConcatenationFusion, ReliabilityGatedFusion, masked_softmax


def test_masked_softmax_sums_to_one_and_zeros_missing():
    scores = torch.tensor([[2.0, 5.0, -1.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.bool)
    weights = masked_softmax(scores, mask)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.equal(weights.masked_select(~mask), torch.zeros(3))


@pytest.mark.parametrize("mask", [[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_reliability_fusion_all_availability_patterns(mask):
    fusion = ReliabilityGatedFusion(12)
    representations = torch.randn(1, 3, 12)
    boolean_mask = torch.tensor([mask], dtype=torch.bool)
    fused, weights = fusion(representations, boolean_mask, torch.rand(1, 3))
    assert fused.shape == (1, 12)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert torch.equal(weights.masked_select(~boolean_mask), torch.zeros(int((~boolean_mask).sum())))


def test_concatenation_fusion_masks_unavailable_vectors():
    fusion = ConcatenationFusion(8)
    representations = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    output, weights = fusion(representations, mask)
    assert output.shape == (2, 8)
    assert torch.equal(weights.masked_select(~mask), torch.zeros(3))


def test_all_missing_fusion_is_rejected():
    fusion = ReliabilityGatedFusion(4)
    with pytest.raises(ValueError, match="all-masked"):
        fusion(torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.bool), torch.ones(1, 3))
