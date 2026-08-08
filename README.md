# Multilingual Multimodal Emotion Recognition

Clean-room PyTorch framework for low-resource multilingual multimodal emotion
recognition. Audio, text, and visual inputs are independent; every run may enable any
non-empty subset without creating modules or requiring caches for disabled modalities.

## Current implementation

- JSONL manifests with explicit availability and leakage validation.
- Speaker-exclusive CREMA-D full (64/14/13 actors) and pilot (8/2/2 actors) protocols.
- Frozen XLS-R audio, Qwen3 text, and SigLIP visual SafeTensor caches.
- Modality projections, optional modality/emotion adapters, optional language/corpus
  routing, masked concatenation, and reliability-aware fusion.
- UAR, macro-F1, accuracy, grouped metrics, early stopping, cache/checkpoint contract
  verification, and deterministic offline tests.
- True unimodal, bimodal, and trimodal execution. Disabled branches have zero fusion
  weight and contribute no model parameters.

Prototypical meta-learning, supervised contrastive learning, face crops, and temporal
visual attention are future controlled experiments. Federated learning and
conversational ERC are outside the current scope.

The two Word documents describe an audio-text paper, while the later implementation
prompt defines a reusable trimodal framework. Their naming and evidence boundaries are
explained in `docs/spec_alignment.md`.

## Environment

Use Python 3.10–3.14 from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,encoders]"
```

Installing dependencies does not download pretrained weights. Encoder commands download
weights only when `--allow-download` is supplied.

## Verify

```powershell
python -m pytest
python scripts/train.py --config configs/experiment/cpu_smoke.yaml
python scripts/evaluate.py --config configs/experiment/cpu_smoke.yaml
```

Synthetic outputs are ignored by Git and explicitly marked `synthetic: true`.

## CREMA-D manifests

With the official dataset at `data/raw/crema_d`:

```powershell
python scripts/build_cremad_manifest.py
```

The approved mapping keeps `ANG/HAP/NEU/SAD` as angry/happy/neutral/sad and excludes
`DIS/FEA` instead of merging them. See `docs/cremad_protocol.md`.

## Frozen encoder caches

Inspect all three extraction contracts without downloading or loading a model:

```powershell
python scripts/cache_audio_embeddings.py --dry-run
python scripts/cache_text_embeddings.py --dry-run
python scripts/cache_visual_embeddings.py --dry-run
```

After explicit approval, a canary followed by resumable pilot extraction is:

```powershell
python scripts/cache_audio_embeddings.py --allow-download --limit 16
python scripts/cache_audio_embeddings.py

python scripts/cache_text_embeddings.py --allow-download --limit 16
python scripts/cache_text_embeddings.py

python scripts/cache_visual_embeddings.py --decode-check --limit 16
python scripts/cache_visual_embeddings.py --allow-download --limit 16
python scripts/cache_visual_embeddings.py
```

See `docs/audio_cache_protocol.md`, `docs/text_cache_protocol.md`, and
`docs/visual_cache_protocol.md` for exact pooling, revision, dtype, and limitations.

## Real cached experiments

The existing trimodal sanity run remains an engineering check:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_sanity.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_sanity.yaml
```

The reusable framework's supervised reliability experiment is named descriptively:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_supervised_reliability.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_supervised_reliability.yaml
```

`cremad_pilot_p2.yaml` remains only as a deprecated alias so old commands do not break.
It is not the audio-text paper's P2, which requires future Prototypical Networks.

To verify that the paper-oriented audio-text path is independent from visual data, run:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_baseline_audio_text_concat.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_baseline_audio_text_concat.yaml
```

That configuration contains only audio/text dimensions and cache sources. It is still a
CREMA-D pilot diagnostic, not multilingual paper evidence.

Historical representation diagnostics remain available:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_diag_audio.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_text.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_visual.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_audio_visual_concat.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_trimodal_concat.yaml
```

These historical configurations include adapters and routing, so they are diagnostics,
not parameter-fair paper baselines. All real pilot outputs remain `paper_ready: false`.

## Architecture and safeguards

Each enabled frozen-encoder vector is projected to `d_model`. Modality adapters, a
shared/separate emotion adapter, and language/corpus routes are independently
configurable. Fusion preserves canonical audio/text/visual weight columns even when a
branch is disabled.

- No unavailable modality is replaced with another modality's feature.
- Original-language text is preserved; translation is not mandatory preprocessing.
- Speaker and source-video grouping precede splitting.
- Incompatible labels require an explicit mapping.
- Language–corpus confounding is reported, never interpreted as a language effect.
- CREMA-D has only twelve fixed official prompts and one English corpus.
- Current pilot results and single-seed diagnostics are never paper-ready evidence.
- Datasets, source references, caches, checkpoints, outputs, credentials, and model
  weights remain private and untracked.

The complete correction list is in `docs/code_audit_20260808.md`.
