# Implementation plan

## Verified supervised framework

1. Define an explicit JSONL manifest and four-label mapping.
2. Validate metadata, files, identities, all-split leakage, imbalance, availability, and
   language–corpus confounding.
3. Cache frozen audio/text/visual embeddings under hash-bound contracts.
4. Instantiate projections and optional adapters only for enabled modalities.
5. Apply a shared, separate, or disabled emotion-adapter layer.
6. Optionally route language- and corpus-specific residual adapters.
7. Compare subset-aware masked concatenation and reliability-aware fusion.
8. Train, save a reproducibility bundle, reload only a matching checkpoint, and run
   deterministic offline tests.

## Completed real-data engineering

- Speaker-exclusive CREMA-D full and pilot manifests.
- Verified 648-clip XLS-R audio cache, 648-record/12-transcript Qwen3 text cache, and
  648-clip SigLIP visual cache.
- Strict enabled-modality cache assembly and checkpoint/cache contract verification.
- A true audio-text baseline that requires no visual dimensions, indexes, contracts, or
  tensors.
- Historical seed-17 audio, text, visual, audio-visual, and trimodal diagnostics.
- Clear separation between the audio-text paper specification and trimodal framework.

## Completed supervised framework components

- Executable B1 unimodal, B2 bimodal, B3 concatenation, B4 metadata, B5
  parameter-matched, P1, P2, and P3 configurations.
- Audio resampling/downmixing, chunk policies, hidden-layer selection, masked mean, and
  deterministic attentive-statistics pooling.
- Full-frame mean, optional face-crop mean, and trainable frame-level temporal attention.
- AMP/autocast, gradient accumulation, registered adapter/fusion/modality ablations,
  missing-modality stress tests, and three-seed aggregation.
- Metrics sliced by language, corpus, modality pattern, transcript source, country,
  region, and accent when those fields exist.

## Completed episodic framework engineering

- Balanced language/corpus episode sampling with no support/query utterance reuse.
- Optional speaker-disjoint support/query sampling for validation and evaluation.
- Cosine and squared-Euclidean Prototypical Networks.
- Optional supervised contrastive loss over the pre-routing shared representation.
- Zero-shot classifier evaluation plus 1-/5-/10-shot prototype evaluation.
- Episode-level mean, standard deviation, 95% confidence interval, and provenance.
- Metadata embeddings as the B4 implementation component.

## Remaining paper work

- Add legally obtained English/Mandarin corpora and corpus converters; no current code
  fabricates or downloads unapproved research data.
- Establish speaker/source-video groups and actual source/target corpus tasks.
- Extract the optional ablation caches, then execute the registered matrix for seeds
  17/23/41 and report mean/std/CI.
- Add paired gold/ASR transcripts before executing that specific ablation.

Federated learning and conversational ERC remain separate future projects.
