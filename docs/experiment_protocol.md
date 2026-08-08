# Experiment protocol

## Data contract

Run manifest validation before cache extraction or training. Invalid availability,
paths, splits, duplicate identities, source-video overlap, or speaker overlap across any
split is fatal. Keep the manifest and enabled cache-contract hashes with every result.
Country, region, and accent remain nullable descriptive fields and are never inferred.

## Reproducible training

- Use frozen encoders and immutable cached embeddings.
- Seed Python, NumPy, PyTorch, CUDA, and DataLoader generators.
- Save resolved YAML, labels, versions, git/source hash, manifest/cache hashes, parameter
  counts, checkpoint path, and metrics.
- Select checkpoints by validation UAR only; evaluate test speakers after selection.
- Report synthetic and diagnostic runs separately from scientific experiments.
- Refuse evaluation when checkpoint model, label, manifest, or cache contracts differ.

## Metrics and slices

Primary: UAR. Secondary: macro-F1, accuracy, per-class recall, and confusion matrix.
Report overall, per language, per corpus, and per modality pattern. For reliability
fusion, report average weights for the same slices and verify every unavailable or
disabled modality has exactly zero weight.

## Baselines and ablations

`configs/baseline/baselines.yaml` is a research registry, not an automatic preset
expander. Every executed baseline needs a complete experiment YAML. Main comparisons
should cover unimodal branches, all bimodal pairs, trimodal concatenation, hierarchical
adapters, and reliability fusion with explicit parameter counts. Later controlled
ablations cover metadata embeddings, parameter-matched adapters, visual pooling/crops,
transcript source, and missing-modality stress tests.

The paper audio-text and reusable trimodal framework tracks use different stage names;
see `docs/spec_alignment.md`.

## Real-data split policy

All utterances from a speaker and original source video stay in one split. Dialogue
media must additionally group by dialogue/episode. Language–corpus confounding is
reported and tested through held-out corpora, never presented as a clean language
causal effect.
