# Code audit — 2026-08-08

## Corrected defects

- Disabled modalities no longer create projections, adapters, router work, fusion
  scorers, or concatenation parameters.
- A run now requires cache indexes and contracts only for enabled modalities. An
  audio-text experiment can omit every visual path, dimension, and cache source.
- Fusion still returns canonical audio/text/visual weight columns, with exact zero for
  disabled or unavailable modalities.
- Evaluation verifies the label mapping, model constructor contract, manifest hash, and
  all enabled cache-contract hashes before loading model weights.
- Configuration, label, output, checkpoint, manifest, cache-index, and tensor paths are
  constrained to the selected project root.
- Synthetic caches and checkpoints use PyTorch's restricted `weights_only` loader rather
  than unrestricted pickle loading.
- Manifest validation now treats invalid splits, availability/path contradictions,
  invalid confidence/duration, root escapes, and speaker overlap across any pair of
  splits as blocking errors.
- Previously ignored `mixed_precision: true` and gradient accumulation values now fail
  explicitly instead of creating a misleading run. The repaired `small_gpu.yaml` is an
  executable synthetic GPU smoke configuration.
- Audit metadata now records language, corpus, speaker, label, transcript, manifest
  availability, active availability, and enabled-cache summaries without hard-coded
  CREMA-D limitations.

## Intentionally not implemented

- Prototypical episodic training and supervised contrastive learning;
- face crops and temporal visual attention;
- metadata-embedding and parameter-matched adapter ablations;
- claims that any current pilot result is paper-ready.

These are future research components, not defects in the verified cached supervised
pipeline.
