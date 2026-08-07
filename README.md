# Multilingual Multimodal Emotion Recognition

Clean-room PyTorch framework for **Language-Aware Meta-Adapters for Low-Resource
Multilingual Multimodal Emotion Recognition**. The system supports independent audio,
text, and visual streams, missing modalities, cached frozen encoders, lightweight
adapters, and reliability-aware fusion.

## Implemented

- JSONL manifests with explicit modality availability and leakage validation.
- Speaker-exclusive CREMA-D full (64/14/13 actors) and pilot (8/2/2 actors) protocols.
- Frozen XLS-R audio caching with 16 kHz waveform checks and masked-mean pooling.
- Frozen Qwen3 text caching with exact-transcript deduplication and confounding audit.
- Frozen SigLIP visual-cache framework with PyAV decoding, eight uniformly sampled
  full frames, official image preprocessing, mean pooling, visual quality metadata, and
  a verified 648-clip CREMA-D pilot aligned with the audio/text caches.
- Three projections, modality adapters, a shared emotion adapter, and language/corpus
  residual routing.
- Concatenation and reliability-aware gated fusion with exact zero weight for missing
  modalities.
- UAR, macro-F1, accuracy, checkpoints, deterministic synthetic smoke runs, and tests.

Prototypical meta-learning, face-crop features, and temporal visual attention remain
later controlled experiments. Federated learning and conversational ERC are outside the
current scope.

## Environment

Use Python 3.10–3.14. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,encoders]"
```

Installing dependencies does not itself download model weights. Pretrained weights are
loaded only by an extraction command with `--allow-download`.

## Verify the implementation

```powershell
python -m pytest
python scripts/train.py --config configs/experiment/cpu_smoke.yaml
python scripts/evaluate.py --config configs/experiment/cpu_smoke.yaml
```

Synthetic outputs are written under `outputs/`, ignored by Git, and explicitly marked
`"synthetic": true`.

## CREMA-D manifests

With the official dataset at `data/raw/crema_d`, build the approved manifests:

```powershell
python scripts/build_cremad_manifest.py
```

The protocol maps intended filename labels `ANG/HAP/NEU/SAD` to angry/happy/neutral/sad
and excludes `DIS/FEA` without merging. See `docs/cremad_protocol.md`.

## Frozen encoder caches

All extractors default to the 648-clip CREMA-D pilot. Inspect their resolved contracts
without loading a model:

```powershell
python scripts/cache_audio_embeddings.py --dry-run
python scripts/cache_text_embeddings.py --dry-run
python scripts/cache_visual_embeddings.py --dry-run
```

### Audio

The pinned `facebook/wav2vec2-xls-r-300m` extractor caches one float32 1024-D vector per
clip. A canary and resumable pilot extraction are:

```powershell
python scripts/cache_audio_embeddings.py --allow-download --limit 16
python scripts/cache_audio_embeddings.py
```

See `docs/audio_cache_protocol.md`.

### Text

The pinned `Qwen/Qwen3-Embedding-0.6B` extractor caches normalized float32 1024-D
vectors. Exact transcripts are deduplicated: CREMA-D's 648 pilot records contain only
12 distinct prompted sentences.

```powershell
python scripts/cache_text_embeddings.py --allow-download --limit 16
python scripts/cache_text_embeddings.py
```

See `docs/text_cache_protocol.md` for the prompt-frequency confounding limitation.

### Visual

Before loading SigLIP, validate real FLV decoding and uniform frame sampling:

```powershell
python scripts/cache_visual_embeddings.py --decode-check --limit 16
```

After explicit approval for the approximately 813 MB checkpoint, run a 16-clip canary,
then resume the complete pilot from locally cached weights:

```powershell
python scripts/cache_visual_embeddings.py --allow-download --limit 16
python scripts/cache_visual_embeddings.py
```

The baseline uses `google/siglip-base-patch16-224`, eight full RGB frames, and mean
pooling into one float32 768-D clip vector. Face crop is intentionally disabled until a
later ablation. See `docs/visual_cache_protocol.md`.

## Architecture

Each cached frozen-encoder vector is projected to `d_model`, passed through a modality
adapter and shared emotion adapter, then receives language- and corpus-specific
residuals. Fusion is either masked concatenation or a quality-conditioned gate over
audio/text/visual. Any subset of the three modalities is supported, including an
audio–text-only paper configuration.

## Research safeguards

- No missing modality is replaced with fabricated features.
- Original Mandarin text is preserved; translation is not mandatory preprocessing.
- Speaker and original-video grouping precede train/test splitting.
- Incompatible labels require explicit mappings.
- Language–corpus confounding is reported rather than interpreted as a language effect.
- Full-frame visual features may encode identity/background shortcuts; face crops are a
  later controlled comparison.
- Reference documents, datasets, model weights, caches, checkpoints, and credentials are
  ignored by Git.
