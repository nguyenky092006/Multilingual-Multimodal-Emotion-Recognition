# EmotionTalk full audio-text experiment protocol

## Status

The full four-class Mandarin manifest contains 16,413 utterances with 11/4/4
train/validation/test speakers and no speaker or source-group overlap. Frozen cache
extraction is complete for XLS-R audio and Qwen3 text. Every configuration below remains
an engineering diagnostic until repeated seeds and the English-Mandarin cross-corpus
protocol are frozen.

## Verified cache contracts

- Audio: `data/cache/audio/emotiontalk_full_xlsr300m_chunk12s`; masked-mean XLS-R,
  12-second non-overlapping chunks, duration-weighted chunk aggregation, float32 cache.
- Text: `data/cache/text/emotiontalk_full_qwen3_embedding_0.6b`; original Chinese,
  Qwen3 last-token pooling, normalized float32 cache, exact transcript deduplication.
- Both embeddings are 1024-dimensional and bound to
  `data/manifests/emotiontalk_full.jsonl` by manifest/summary hashes.
- No visual cache, tensor, or dimension is used.

## Registered comparisons

1. `emotiontalk_full_baseline_audio.yaml`
2. `emotiontalk_full_baseline_text.yaml`
3. `emotiontalk_full_baseline_audio_text_concat.yaml`
4. `emotiontalk_full_p1_reliability.yaml`
5. `emotiontalk_full_p2_meta.yaml`
6. `emotiontalk_full_p3_meta.yaml`

The supervised configurations use inverse-frequency class weights because the full
manifest is strongly imbalanced. P2 and P3 retain that weighting for their global
classification term; episodic support/query batches are class balanced. P2 adds the
5-shot Prototypical objective. P3 differs by enabling supervised contrastive loss at
weight 0.1. Validation and test episodes enforce speaker-disjoint support/query sets and
report zero-/1-/5-/10-shot metrics.

## Interpretation boundary

EmotionTalk alone is Mandarin and one corpus. Routed adapters are executable, but their
single route is not evidence of language/corpus adaptation. A CREMA-D/EmotionTalk
comparison confounds language with corpus and must say so explicitly. Pilot and
single-seed full results are not paper-ready; use at least seeds 17/23/41 after configs
and parameter budgets are frozen.
