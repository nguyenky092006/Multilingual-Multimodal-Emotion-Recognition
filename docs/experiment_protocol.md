# Experiment protocol

## Data contract

Run manifest validation before cache extraction or training. Serious sample, path,
source-video, or speaker leakage is fatal. Keep a manifest hash with every result.
Country, region, and accent are descriptive nullable fields, not inferred labels.

## Reproducible training

- Use frozen encoders and immutable cached embeddings.
- Seed Python, NumPy, PyTorch, CUDA, and DataLoader generators.
- Save the resolved YAML, label mapping, package versions, git hash, manifest/cache hash,
  parameter counts, checkpoint path, and metrics.
- Select checkpoints by validation macro recall only. Evaluate the held-out test set once
  per final seed/configuration.
- Report synthetic runs separately from scientific experiments.

## Metrics and slices

Primary: UAR (macro recall). Secondary: macro-F1, accuracy, per-class recall, and a
confusion matrix. Report overall, per language, per corpus, and per modality pattern.
For reliability fusion, report average weights for the same slices and verify unavailable
modalities have exactly zero weight.

## Baselines and ablations

The preset matrix is in `configs/baseline/baselines.yaml`. Main runs should include the
three unimodal baselines, all bimodal concatenations, trimodal concatenation, routed
adapters, and reliability fusion. Later ablations remove each adapter/router/gate,
compare parameter-matched adapters, visual pooling/crop choices, transcript source, and
stress missing modalities.

## Real-data split policy

All utterances from a speaker and original source video stay in one split. With dialogue
media, group by source dialogue/episode as well. Any unavoidable language–corpus
confounding is documented and evaluated through corpus-held-out slices, never presented
as a clean language causal effect.
