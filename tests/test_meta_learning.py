from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from mmer.adapters import MetadataConditioner
from mmer.config import load_yaml
from mmer.data.synthetic import make_synthetic_dataset
from mmer.meta import (
    EpisodeSampler,
    prototypical_loss,
    supervised_contrastive_loss,
)
from mmer.meta.engine import evaluate_meta_episodes, train_meta_epoch
from mmer.models import TrimodalEmotionModel
from mmer.meta_runner import _domain_protocol_audit


DIMS = {"audio": 12, "text": 10, "visual": 8}
LABELS = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
ROOT = Path(__file__).resolve().parents[1]


def _network() -> TrimodalEmotionModel:
    return TrimodalEmotionModel(
        DIMS,
        4,
        ["en"],
        ["synthetic_a", "synthetic_b"],
        d_model=16,
        projection_hidden=24,
        adapter_bottleneck=4,
        dropout=0.0,
        enabled_modalities=["audio", "text", "visual"],
    )


def test_prototypical_loss_recovers_separated_classes():
    support = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    query = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
    result = prototypical_loss(support, labels, query, torch.tensor([0, 1]))
    assert result.predictions.tolist() == [0, 1]
    assert torch.isfinite(result.loss)


def test_episode_sampler_is_deterministic_and_has_no_utterance_overlap():
    dataset = make_synthetic_dataset(64, DIMS, seed=7, labels=LABELS, split="test")
    kwargs = dict(
        dataset=dataset,
        n_way=4,
        k_shot=1,
        query_per_class=1,
        episodes=4,
        seed=19,
        task_field="global",
        disjoint_speakers=True,
    )
    first = list(EpisodeSampler(**kwargs))
    second = list(EpisodeSampler(**kwargs))
    assert [
        [sample.sample_id for sample in item.support] for item in first
    ] == [
        [sample.sample_id for sample in item.support] for item in second
    ]
    for episode in first:
        assert not ({item.sample_id for item in episode.support} & {item.sample_id for item in episode.query})
        assert not (set(episode.support_speakers) & set(episode.query_speakers))


def test_meta_epoch_and_episode_evaluation_run_on_cpu():
    dataset = make_synthetic_dataset(64, DIMS, seed=11, labels=LABELS, split="train")
    sampler_args = dict(
        dataset=dataset,
        n_way=4,
        k_shot=1,
        query_per_class=1,
        episodes=2,
        task_field="global",
    )
    network = _network()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    losses = train_meta_epoch(
        network,
        EpisodeSampler(seed=3, disjoint_speakers=False, **sampler_args),
        optimizer,
        torch.device("cpu"),
        distance="cosine",
        prototype_temperature=0.1,
        lambda_classification=1.0,
        lambda_episode=1.0,
        lambda_supcon=0.1,
        supcon_temperature=0.1,
    )
    assert all(math.isfinite(value) for value in losses.values())
    metrics = evaluate_meta_episodes(
        network,
        EpisodeSampler(seed=4, disjoint_speakers=True, **sampler_args),
        torch.device("cpu"),
        num_classes=4,
        distance="cosine",
        prototype_temperature=0.1,
    )
    assert metrics["episodes"] == 2
    assert metrics["support_query_sample_overlap_detected"] is False
    assert metrics["support_query_speaker_overlap_episodes"] == 0
    assert 0.0 <= metrics["uar"]["ci95_low"] <= metrics["uar"]["ci95_high"] <= 1.0


def test_supervised_contrastive_and_metadata_unknown_route_are_finite():
    features = torch.randn(8, 6, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(features, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()

    conditioner = MetadataConditioner(6, ["en"], ["crema-d"])
    output, stats = conditioner(torch.randn(2, 6), ["en", "zh"], ["crema-d", "new"])
    assert output.shape == (2, 6)
    assert stats["language_usage"] == {"en": 1, "unknown": 1}
    assert stats["corpus_usage"] == {"crema-d": 1, "unknown": 1}


@pytest.mark.parametrize(
    ("name", "supcon"),
    [
        ("cremad_full_framework_p2_meta.yaml", 0.0),
        ("cremad_full_framework_p3_meta.yaml", 0.1),
    ],
)
def test_full_meta_configs_have_speaker_disjoint_k_shot_contract(name: str, supcon: float):
    config = load_yaml(ROOT / "configs" / "experiment" / name)
    assert config["data"]["enabled_modalities"] == ["audio", "text"]
    assert config["meta"]["n_way"] == 4
    assert config["meta"]["evaluation_k_shots"] == [1, 5, 10]
    assert config["meta"]["disjoint_speakers_test"] is True
    assert config["meta"]["lambda_supcon"] == supcon
    assert config["diagnostic"] is True


def test_unseen_corpus_protocol_is_verified_or_rejected_explicitly():
    audit = {
        "split_corpus_counts": {
            "train": {"source-a": 10, "source-b": 10},
            "validation": {"source-a": 4},
            "test": {"target": 8},
        }
    }
    result = _domain_protocol_audit({"require_unseen_test_corpus": True}, audit)
    assert result["unseen_test_corpus_verified"] is True
    audit["split_corpus_counts"]["test"] = {"source-a": 8}
    with pytest.raises(ValueError, match="disjoint train/test"):
        _domain_protocol_audit({"require_unseen_test_corpus": True}, audit)
