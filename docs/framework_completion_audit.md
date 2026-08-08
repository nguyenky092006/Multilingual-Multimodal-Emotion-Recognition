# Framework completion audit

This document separates implemented research code from evidence that still requires
data or compute. “Implemented” means tested and callable; it does not mean the associated
experiment has already been run or supports a scientific claim.

## Implemented and tested

| Requirement | Implementation |
| --- | --- |
| Independent audio/text/visual subsets | Cache loader and model instantiate only enabled branches; canonical masks remain comparable. |
| Frozen encoders | XLS-R, Qwen3 with smaller multilingual fallback, and SigLIP cache scripts with immutable SafeTensor contracts. |
| Encoder aggregation | Audio masked mean/statistics/energy-attentive statistics; visual frame mean or trainable temporal attention; optional face crop with fallback. |
| Adapters and routing | Modality, shared/separate emotion, language, corpus, and independently removable language/corpus routes. |
| Fusion | Parameterized concatenation and availability-masked reliability fusion; unavailable weights are exactly zero. |
| Few-shot learning | Balanced Prototypical episodes, zero-/1-/5-/10-shot evaluation, optional supervised contrastive loss, and disjoint-support/query controls. |
| Baselines | Executable B1, B2, B3, B4, B5, P1, P2, and P3 registries/configurations. |
| Ablations | Modality, adapter, router, gate, meta-learning, encoder pooling/crop, parameter matching, and missing-modality stress paths. |
| Training controls | Deterministic seeds, AMP/autocast, gradient accumulation, early stopping, checkpoint contract verification, and three-seed orchestration. |
| Reporting | UAR/macro-F1/accuracy, confusion/per-class metrics, metadata slices, gate summaries, episode CI, parameter counts, and seed mean/std/CI. |
| Protocol safety | Speaker/source leakage checks, enabled-cache validation, unseen-test-corpus enforcement, root-constrained paths, and no fabricated missing data. |

## Data- or execution-blocked

| Item | Why it is not complete evidence |
| --- | --- |
| English-Mandarin comparison | EmotionTalk audio/text is present and has a tested Mandarin converter, but its frozen caches and parameter-fair English-Mandarin runs have not been executed. Language remains confounded with corpus. |
| Unseen-corpus adaptation result | Requires at least one held-out target corpus. The runner rejects overlap when `require_unseen_test_corpus` is enabled. |
| Gold-vs-ASR ablation | Requires paired gold and ASR transcripts for the same samples; none are fabricated. |
| Full visual ablation results | Frame-level and face-crop caches must be extracted under their dedicated contracts before training. |
| Paper tables | B1-B5/P1-P3 and ablations must be frozen and run for at least seeds 17/23/41; single-seed CREMA-D results remain engineering diagnostics. |

Federated learning and conversational emotion recognition are explicitly outside this
independent-utterance framework. They can be separate later projects without changing
the current evidence boundary.
