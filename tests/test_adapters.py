from __future__ import annotations

import torch

from mmer.adapters import AdapterRouter, ResidualAdapter


def test_residual_adapter_preserves_dimensions():
    adapter = ResidualAdapter(16, 4)
    inputs = torch.randn(5, 16)
    assert adapter(inputs).shape == inputs.shape


def test_router_handles_mixed_language_and_corpus_batches():
    router = AdapterRouter(16, 4, ["en", "zh"], ["a", "b"])
    output, stats = router(torch.randn(4, 16), ["en", "zh", "en", "zh"], ["a", "a", "b", "b"])
    assert output.shape == (4, 16)
    assert stats["language_usage"] == {"en": 2, "zh": 2}
    assert stats["corpus_usage"] == {"a": 2, "b": 2}


def test_router_maps_unknown_routes_explicitly():
    router = AdapterRouter(8, 2, ["en"], ["known"])
    _, stats = router(torch.randn(2, 8), ["fr", "unknown"], ["new", "unknown"])
    assert stats["language_usage"] == {"unknown": 2}
    assert stats["corpus_usage"] == {"unknown": 2}
    assert isinstance(stats["collapse_detected"], bool)


def test_router_supports_language_and_corpus_ablation_independently():
    language_only = AdapterRouter(
        8, 2, ["en"], ["known"], use_corpus_adapters=False
    )
    _, language_stats = language_only(torch.randn(2, 8), ["en", "new"], ["known", "new"])
    assert language_stats["language_usage"] == {"en": 1, "unknown": 1}
    assert language_stats["corpus_usage"] == {}

    corpus_only = AdapterRouter(
        8, 2, ["en"], ["known"], use_language_adapters=False
    )
    _, corpus_stats = corpus_only(torch.randn(2, 8), ["en", "new"], ["known", "new"])
    assert corpus_stats["language_usage"] == {}
    assert corpus_stats["corpus_usage"] == {"known": 1, "unknown": 1}
