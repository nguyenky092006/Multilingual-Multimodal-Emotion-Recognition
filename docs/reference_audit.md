# Reference audit

Audit date: 2026-08-06. The four original reference files were read from the ignored
`references/source/` directory and were not modified.

## SER-Fuse (ACM MM 2023)

SER-Fuse is an audio–text speech-emotion system, not a trimodal MMER implementation.
It combines BERT text representations with ECAPA-TDNN/x-vector/d-vector speech
representations and an MLP fusion classifier. Its ECES experiment combines English
and Mandarin ESD-derived examples, but language and collection can be confounded and
the reported random split does not establish speaker-independent evaluation. The
paper is useful for failure modes, feature-cache organization, and metric comparison;
its encoders and two-branch fusion are not copied.

Official repository: https://github.com/nhattruongpham/SER-Fuse

At audit time the public repository exposed three notebooks and supporting artifact
directories but no top-level license. No source was copied or adapted.

## LaERC-S (COLING 2025)

LaERC-S is text-only emotion recognition in conversation with LLM instruction tuning,
speaker characteristics, dialogue history, and LoRA. These objectives conflict with
this repository's independent-clip definition. No history, prompt, or ERC logic is
reused. Generic ideas about explicit configuration and reproducibility are applicable.

Official repository: https://github.com/bigcat-1/LaERC-S

At audit time it contained a minimal README and no implementation or top-level license.
There was consequently nothing to adapt.

## Research summary and blueprint

The two-page summary and detailed blueprint propose frozen audio/text encoders,
shared/specific adapters, a reliability gate, and later episodic ProtoNet training.
Both documents are fundamentally audio–text plans. This implementation retains their
parameter-efficient adapter hypothesis but expands the software contract to three
real modalities. ProtoNet is deliberately deferred until cached trimodal supervised
training is stable.

## Encoder audit

- Audio: `facebook/wav2vec2-xls-r-300m` is the frozen default. Cache extraction will
  use mono 16 kHz input, masked pooling, a selectable layer, and chunking. Extraction
  is specified but intentionally guarded in this iteration.
- Text: the verified identifier is `Qwen/Qwen3-Embedding-0.6B`. Its official model card
  identifies an Apache-2.0 0.6B, 1024-dimensional multilingual embedding model with
  configurable output dimensions. The documented detailed inference uses last-token
  pooling with attention-mask-aware handling and recommends current Transformers
  support. The original-language transcript is retained. A smaller multilingual
  Sentence-Transformers model is configurable, not silently substituted.
- Visual: the initial configuration chooses a frame-based SigLIP encoder because its
  official Transformers implementation supports independent image encoding and cached
  embeddings at a practical frame count. Uniform eight-frame sampling and masked mean
  pooling form the baseline. DINOv2, CLIP, VideoMAE, facial encoders, temporal attention,
  and face crops remain explicit ablations. Face detection is never mandatory.

No pretrained weights were downloaded during this audit.

## Direct reuse decision

No external code is reused. All interfaces, adapters, fusion, validation, metrics, and
synthetic data code in this repository are clean-room implementations. See `NOTICE.md`.

