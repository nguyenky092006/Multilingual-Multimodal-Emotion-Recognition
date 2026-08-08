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

## Next supervised work

- Add legally obtained English/Mandarin data and corpus converters.
- Establish speaker/source-video group splits for every new corpus.
- Add complete parameter-fair experiment YAMLs for all unimodal/bimodal/trimodal and
  adapter/fusion comparisons.
- Run at least three seeds, then aggregate confidence intervals.
- Add face-crop, temporal pooling, transcript-source, and missing-modality ablations.

## Later paper/meta-learning work

- Add episodic samplers and Prototypical Networks for zero-/1-/5-/10-shot evaluation.
- Add the supervised contrastive extension only after ProtoNet is verified.
- Run within-corpus, cross-corpus, and unseen-corpus protocols.

ProtoNet is intentionally outside the current supervised implementation. Federated
learning and conversational ERC remain separate future projects.
