# Real cached-embedding training protocol

## Purpose

This stage connects the verified CREMA-D manifest to the independently cached XLS-R,
Qwen3, and SigLIP representations. Training loads only small float32 SafeTensors; the
three pretrained encoders are never loaded into training memory.

The first run is a **pipeline sanity experiment**, not a paper result. CREMA-D is one
English corpus, the pilot contains only 12 actors, the text stream contains only twelve
fixed official prompts, and one seed cannot support a research conclusion.

## Strict cache assembly

Before constructing a dataset, the loader verifies:

- the manifest and all configured paths remain inside the project root;
- sample IDs are unique and speakers do not cross train/validation/test splits;
- each modality index contains exactly the manifest records marked available;
- cache contracts identify the expected modality, float32 dtype, model revision, and
  manifest hash (directly or through the text manifest summary);
- every SafeTensor contract hash and embedding dimension matches its index/config;
- every embedding is finite;
- train, validation, and test splits all exist.

The verified CREMA-D pilot resolves to 432 training, 108 validation, and 108 test clips.
It loads 648 audio tensors, 12 deduplicated text tensors, and 648 visual tensors.

## Quality policy

`validated_metadata_v1` deliberately uses only values with a direct bounded meaning:

- audio: `1 - clipping_fraction`;
- text: ASR confidence when provided, `1.0` for verified gold/manual/official prompts,
  and `0.0` when a non-gold transcript has no confidence;
- visual: valid sampled frames divided by requested frames.

Availability remains a separate mask. This policy does not invent face-detection,
brightness, sharpness, or ASR scores. In the complete CREMA-D pilot, text and visual
quality are both 1.0 and audio quality is near 1.0, so this corpus alone cannot validate
the quality-metadata hypothesis. The learned gate can still use representations and
availability, but reliability claims require noisier/missing-modality corpora later.

## Safe execution order

First run the three-epoch sanity configuration:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_sanity.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_sanity.yaml
```

Only after checking finite loss, checkpoint reload, split counts, held-out metrics, and
fusion weights should the longer pilot be run:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_p2.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_p2.yaml
```

Both configurations are marked `pilot: true`; their summaries and metrics also set
`synthetic: false` and `paper_ready: false`. Outputs are written beneath
`outputs/cremad_pilot/` and ignored by Git.

## Sanity result and diagnostic gate

The three-epoch seed-17 sanity run completed successfully, including checkpoint reload.
Its held-out test UAR was 0.3661 and macro-F1 was 0.3020. Neutral recall was 0.0, while
the learned gate assigned 74.9% mean weight to visual. These numbers validate plumbing,
not the proposed method, and motivate diagnostics before a longer P2 run.

Run the representation diagnostics separately:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_diag_audio.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_text.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_visual.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_audio_visual_concat.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_trimodal_concat.yaml
```

Then evaluate their best validation checkpoints with the matching configuration and
`scripts/evaluate.py`. These configurations retain the current adapters/router and are
therefore labelled diagnostics, not parameter-fair B1/B3 paper baselines. Their purpose
is to identify encoder signal and determine whether the reliability gate caused the
early visual dominance.

The first seed-17 diagnostic pass produced the following held-out test results. These
are engineering diagnostics on two test speakers, not paper-ready estimates:

| Enabled modalities | Fusion | Test UAR | Macro-F1 |
| --- | --- | ---: | ---: |
| text | unimodal | 0.2827 | 0.1607 |
| audio | unimodal | 0.4732 | 0.4245 |
| audio + visual | fixed concat | 0.4717 | 0.4551 |
| visual | unimodal | 0.5193 | 0.5062 |
| audio + text + visual | fixed concat | 0.5104 | 0.5079 |

Text is nearly uninformative because CREMA-D reuses twelve fixed prompts across
emotion labels. Fixed trimodal concatenation improved validation UAR to 0.6533 but did
not beat visual-only on held-out test speakers. Audio-visual concat reached 0.6652
validation UAR but only 0.4717 test UAR, so removing text did not improve held-out
generalization. The large validation/test gaps show that the two-speaker pilot test is
too variable for selecting the fusion design. Do not run the longer gated P2 as a
paper-facing experiment until the larger speaker split and repeated seeds are ready.

Training metadata stores a SHA-256 snapshot of all Python sources, scripts, the exact
experiment YAML, and `pyproject.toml`. It also flags zero-recall classes and fusion
weights above 70%, so experiments made from an uncommitted worktree are still
distinguishable. Paper-oriented runs should nevertheless start from a clean commit.

## Interpretation limits and next comparisons

The P2 pilot exercises projections, modality adapters, the shared emotion adapter,
language/corpus routes, reliability fusion, modality dropout, class weights, early
stopping, checkpoint reload, and grouped metrics. It does not isolate their effects.

Paper-oriented work must next add parameter-fair configurations and repeated seeds for
audio-only, text-only, visual-only, bimodal pairs, trimodal concatenation, and routed
adapter/gated-fusion ablations. Language and corpus adapters require additional legally
obtained corpora, especially Mandarin data, before their hypotheses can be evaluated.
