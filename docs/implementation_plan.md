# Implementation plan

## Iteration 1 — implemented

1. Define an explicit JSONL manifest and four-label YAML mapping.
2. Validate metadata, files, duplicate identities, split leakage, imbalance, modality
   availability, and language–corpus confounding before training.
3. Generate deterministic synthetic cached audio/text/visual embeddings without models.
4. Project each modality into a common dimension and apply modality-specific adapters.
5. Apply a shared emotion adapter (or parameter-matched separate adapters).
6. Route language- and corpus-specific residual adapters for mixed batches, including
   unknown routes, and log usage/output norms.
7. Compare masked concatenation with reliability-aware gated fusion.
8. Train an offline classifier, compute UAR/macro-F1/accuracy, save a reproducibility
   bundle, reload the checkpoint, and run offline CPU tests.

## Iteration 2 — in progress with explicit approval and real data

- Implemented frozen XLS-R waveform cache extraction with pinned SafeTensor weights,
  masked-mean pooling, source hashes/quality metadata, atomic writes, resume, and a
  verified 648-clip CREMA-D pilot.
- Implemented frozen Qwen3 text cache extraction with pinned SafeTensor weights,
  exact-text deduplication, float32 normalization, per-manifest indexes,
  prompt/label-confounding audit, resume, and a verified 648-sample/12-transcript pilot.
- Implemented the frozen SigLIP visual-cache framework with pinned SafeTensor weights,
  PyAV decoding, eight uniform full frames, mean pooling, quality metadata, atomic
  writes, strict resume, offline fake-model tests, and a verified 648-clip CREMA-D pilot.
  Its sample IDs align exactly with the verified audio and text pilot caches.
- Create corpus converters only for datasets the user has obtained legally.
- Establish speaker/source-video group splits and run within-/cross-corpus baselines.

## Iteration 3 — after supervised stability

- Add episodic samplers and Prototypical Networks for zero-/1-/5-/10-shot evaluation.
- Add temporal visual attention, face-crop, gold-vs-ASR, and missing-modality ablations.
- Run at least three seeds and aggregate confidence intervals.

Federated learning and ERC remain separate future research projects.
