# Cached-embedding meta-learning protocol

## What is implemented

The episodic runner keeps XLS-R, Qwen3, and SigLIP outside the training process. It
trains only the cached-feature projections, lightweight adapters, fusion module, and
classifier. P2 adds a Prototypical Network objective to P1. P3 adds an optional
supervised contrastive objective over the shared representation before
language/corpus-specific routing.

Every episode is balanced by class and belongs to one configured task key (`language`,
`corpus`, `language_corpus`, or `global`). A sample ID cannot appear in both support and
query. Evaluation defaults to different speakers for support and query; the manifest's
speaker-exclusive train/validation/test split supplies the outer isolation boundary.

## CREMA-D engineering run

From the repository root, after the full audio and text caches have been verified:

```powershell
python scripts/train.py --config configs/experiment/cremad_full_framework_p3_meta.yaml
python scripts/evaluate.py --config configs/experiment/cremad_full_framework_p3_meta.yaml
```

The evaluation JSON contains a zero-shot classifier result and speaker-disjoint
prototype results for 1, 5, and 10 shots. Each K-shot result includes episode-level
records, mean, sample standard deviation, and a normal-approximation 95% confidence
interval. The top-level UAR, macro-F1, and accuracy are the configured primary K-shot
mean, currently 5-shot.

P1 and P2 can be run with the corresponding files:

```powershell
python scripts/train.py --config configs/experiment/cremad_full_framework_p1.yaml
python scripts/train.py --config configs/experiment/cremad_full_framework_p2_meta.yaml
```

## Evidence boundary

CREMA-D contains one English corpus and a small set of prompted transcript strings.
Consequently these configurations are diagnostic (`paper_ready: false`). They test the
complete execution path, not multilingual negative transfer or adaptation to an unseen
corpus. EmotionTalk now supplies an approved Mandarin conversion, but paper-facing
results still require its frozen caches, a held-out target corpus, at least three seeds,
and parameter-fair baselines.
