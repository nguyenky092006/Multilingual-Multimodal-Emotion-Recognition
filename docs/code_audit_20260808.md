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
- Mixed precision now uses device-appropriate autocast and gradient accumulation handles
  both complete and final partial windows. The repaired `small_gpu.yaml` is executable.
- Audit metadata now records language, corpus, speaker, label, transcript, manifest
  availability, active availability, and enabled-cache summaries without hard-coded
  CREMA-D limitations.

## Completed after the initial audit

- Prototypical episodic training and optional supervised contrastive learning;
- optional face crops and trainable temporal visual attention;
- metadata-embedding B4 and parameter-budget-matched B5;
- registered component/encoder/missing-modality ablations and three-seed orchestration;
- explicit unseen-test-corpus validation and metadata-sliced metrics.

The remaining blockers are datasets or unexecuted experiments, not missing architecture
code. No current CREMA-D-only or single-seed output is promoted to paper-ready evidence.
