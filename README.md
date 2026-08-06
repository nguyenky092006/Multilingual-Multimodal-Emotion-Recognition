# Multilingual Multimodal Emotion Recognition

Clean-room PyTorch framework for **Language-Aware Meta-Adapters for Low-Resource
Multilingual Multimodal Emotion Recognition**. Iteration 1 operates entirely on cached
audio, text, and visual embeddings and includes an offline synthetic smoke experiment.
It does not download datasets or pretrained models and does not report real research
results.

## What is implemented

- JSONL manifest with explicit audio/text/visual availability and nullable location/accent metadata.
- Leakage checks for IDs, media paths, speaker, original video, and transcript duplicates.
- Three projections and modality-specific residual adapters.
- One shared emotion adapter or parameter-matched separate-adapter ablation.
- Batched English/Mandarin/unknown and known/unknown corpus routing with usage/norm logs.
- Masked concatenation and reliability-aware fusion; missing modalities get exactly zero weight.
- Modality dropout that preserves at least one stream.
- Trimodal classifier, UAR, macro-F1, accuracy, per-class recall, and confusion matrix.
- Deterministic synthetic cached embeddings, checkpoint reload, tests, and reproducibility artifacts.

ProtoNet, real encoder extraction, federated learning, and conversational ERC are not
implemented in this iteration. See `docs/implementation_plan.md`.

## Environment

Use Python 3.10–3.14. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch is the only large dependency in the smoke environment. Installing the optional
`encoders` extra does **not** download model weights, but it is not needed for iteration 1.

## Verify the implementation

```powershell
python -m pytest
python scripts/train.py --config configs/experiment/cpu_smoke.yaml
python scripts/evaluate.py --config configs/experiment/cpu_smoke.yaml
```

Generated caches, checkpoint, resolved config, training summary, and evaluation metrics
are written to `outputs/smoke/` and ignored by Git. Every resulting metric file is marked
`"synthetic": true`.

Aggregate multiple completed seeds only after producing their metric files:

```powershell
python scripts/aggregate_results.py outputs/seed17/evaluation_metrics.json outputs/seed23/evaluation_metrics.json outputs/seed41/evaluation_metrics.json
```

## Manifest

The schema is documented in `configs/data/manifest_schema.yaml`; labels are in
`configs/data/labels.yaml`. One UTF-8 JSON object is stored per line. Example:

```json
{"sample_id":"cremad-1001","audio_path":"data/cremad/audio/1001.wav","video_path":"data/cremad/video/1001.flv","transcript":"It's eleven o'clock","emotion":"neutral","speaker_id":"1001","language":"en","corpus":"crema-d","split":"train","duration":2.1,"transcript_source":"gold","asr_confidence":null,"audio_available":true,"text_available":true,"visual_available":true,"country":null,"region":null,"accent":null,"source_video_id":"1001-sentence-01"}
```

Validate before any training or extraction:

```powershell
python scripts/validate_manifest.py data/manifests/train.jsonl --labels configs/data/labels.yaml --data-root .
```

The command exits with code 2 for serious leakage. Use `--skip-file-checks` only while
reviewing metadata before media has been mounted; it does not disable identity/leakage checks.

## Encoder cache commands

The verified iteration-1 configurations are inspectable offline:

```powershell
python scripts/cache_audio_embeddings.py --dry-run
python scripts/cache_text_embeddings.py --dry-run
python scripts/cache_visual_embeddings.py --dry-run
```

Without `--dry-run`, these commands deliberately stop and request approval. Full extractors
are dataset-specific iteration-2 work because XLS-R, Qwen3 Embedding, and SigLIP weights are
large and real media access/licensing must first be established. There is no silent fallback.

## Architecture

For each available modality, a cached frozen-encoder vector is projected to `d_model`,
passed through a modality adapter and shared emotion adapter, then receives language and
corpus residuals. The router maps unseen values to explicit `unknown` adapters. Fusion is
either masked concatenation or a quality-conditioned gate over audio/text/visual. The
classifier is LayerNorm, dropout, and a linear label head.

Reliability quality slots are currently one normalized value per modality in the synthetic
contract. Real cache builders must derive and document them from audio duration/energy,
text availability/ASR confidence, and valid-frame/visual quality measurements.

## Configuration presets

- `configs/experiment/cpu_smoke.yaml`: offline acceptance run.
- `configs/experiment/small_gpu.yaml`: future cached-embedding training preset.
- `configs/baseline/baselines.yaml`: B1-A/T/V, B2 pairs, B3–B5, P1–P3 matrix.
- `configs/encoder/frozen_encoders.yaml`: frozen encoder identifiers and pooling policy.

No committed configuration contains a local Windows dataset path.

## Research safeguards

- ESD is audio plus transcript, not trimodal.
- Original Mandarin text is preserved; translation is not a preprocessing requirement.
- Incompatible emotion labels are rejected until mapped explicitly.
- Speaker and original-video grouping precede train/test splitting.
- Language–corpus confounding is reported and is not treated as a language effect.
- Reference PDFs/DOCX, data, caches, checkpoints, credentials, and private files are ignored.
- External audit and attribution decisions are recorded in `docs/` and `NOTICE.md`.
