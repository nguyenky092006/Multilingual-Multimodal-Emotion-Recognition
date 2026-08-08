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
- Frozen SigLIP visual caching with PyAV decoding, eight uniformly sampled full frames,
  official image preprocessing, mean pooling, and visual quality metadata.
- Strict manifest/cache/index/contract alignment for real cached training.
- Three projections, modality adapters, a shared emotion adapter, and language/corpus
  residual routing.
- Concatenation and reliability-aware gated fusion with exact zero weight for missing
  modalities.
- UAR, macro-F1, accuracy, per-class/group metrics, class weights, early stopping,
  checkpoints, reproducibility metadata, and deterministic offline tests.

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

## Real cached training

Run the short, explicitly non-paper-ready sanity experiment first:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_sanity.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_sanity.yaml
```

After auditing that output, the longer P2 pilot is:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_p2.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_p2.yaml
```

Both use speaker-exclusive 432/108/108 train/validation/test clips and write ignored
artifacts under `outputs/cremad_pilot/`. They are marked `synthetic: false`,
`pilot: true`, and `paper_ready: false`. See `docs/real_cached_training_protocol.md`.

The first sanity run exposed zero neutral recall and strong visual-gate dominance, so
run the diagnostic configurations before the longer P2 pilot:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_diag_audio.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_text.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_visual.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_audio_visual_concat.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_trimodal_concat.yaml
```

These are diagnostic models, not parameter-fair paper baselines. Training summaries
record a source/config snapshot hash and automatically flag zero-recall classes or a
fusion modality exceeding 70% mean weight. The audio-visual diagnostic isolates whether
the fixed-prompt text representation harms held-out generalization.

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
- CREMA-D text has only twelve official prompts and can encode prompt-frequency artifacts.
- Full-frame visual features may encode identity/background shortcuts; face crops are a
  later controlled comparison.
- One English pilot corpus and one seed are never presented as paper-ready evidence.
- Reference documents, datasets, model weights, caches, checkpoints, outputs, and
  credentials are ignored by Git.
