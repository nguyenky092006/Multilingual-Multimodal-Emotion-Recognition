# Real cached-embedding training protocol

## Purpose and evidence boundary

This stage connects a verified manifest to independently cached frozen-encoder
representations. Training loads only float32 SafeTensors; pretrained encoders are never
loaded into training memory.

Current CREMA-D runs are pipeline or representation diagnostics, not paper results.
CREMA-D is one English corpus, the pilot has 12 actors, its text stream contains only
twelve fixed official prompts, and one seed cannot support a research conclusion.

## Strict cache assembly

Before constructing a dataset, the loader verifies:

- all configured paths remain inside the project root;
- sample IDs are unique and speakers do not cross any pair of splits;
- every enabled modality index exactly matches manifest records marked available;
- cache contracts match modality, float32 dtype, dimension, revision, and manifest;
- every SafeTensor contract hash and tensor is finite;
- train, validation, and test splits all exist.

Only enabled modalities require dimensions, indexes, contracts, or tensors. For example,
the audio-text baseline is independent from every visual path and cache. Dataset batches
still expose canonical audio/text/visual mask columns so metrics remain comparable.

The trimodal CREMA-D pilot resolves to 432/108/108 train/validation/test clips and uses
648 audio tensors, 12 deduplicated text tensors mapped to 648 records, and 648 visual
tensors.

## Quality policy

`validated_metadata_v1` uses only directly bounded values:

- audio: `1 - clipping_fraction`;
- text: ASR confidence, `1.0` for verified gold/manual/official prompts, otherwise `0.0`;
- visual: valid sampled frames divided by requested frames.

Availability remains a separate mask. The policy does not invent face-detection,
brightness, sharpness, or ASR scores. CREMA-D text/visual quality is essentially
constant and audio quality is near one, so this corpus cannot validate a reliability
claim by itself.

## Safe execution order

Run the short trimodal sanity configuration first:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_sanity.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_sanity.yaml
```

The longer supervised reliability framework experiment is:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_supervised_reliability.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_supervised_reliability.yaml
```

The former `cremad_pilot_p2.yaml` is only a deprecated alias. The paper-oriented P2/P3
episodic path is implemented by the explicit full framework configs; CREMA-D runs verify
the engine but cannot establish multilingual or unseen-corpus claims. See
`docs/spec_alignment.md` and `docs/meta_learning_protocol.md`.

To test the audio-text path without visual data:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_baseline_audio_text_concat.yaml
python scripts/evaluate.py --config configs/experiment/cremad_pilot_baseline_audio_text_concat.yaml
```

All configurations are `pilot: true`, `synthetic: false`, and `paper_ready: false`.

## Historical diagnostics

The three-epoch seed-17 sanity run completed with test UAR 0.3661 and macro-F1 0.3020.
Neutral recall was zero and the gate placed 74.9% mean weight on visual. These values
validate plumbing only.

Historical diagnostic configurations are:

```powershell
python scripts/train.py --config configs/experiment/cremad_pilot_diag_audio.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_text.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_visual.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_audio_visual_concat.yaml
python scripts/train.py --config configs/experiment/cremad_pilot_diag_trimodal_concat.yaml
```

They retain adapters/router and are not parameter-fair paper baselines. Their held-out
seed-17 engineering results were:

| Enabled modalities | Fusion | Test UAR | Macro-F1 |
| --- | --- | ---: | ---: |
| text | unimodal | 0.2827 | 0.1607 |
| audio | unimodal | 0.4732 | 0.4245 |
| audio + visual | fixed concat | 0.4717 | 0.4551 |
| visual | unimodal | 0.5193 | 0.5062 |
| audio + text + visual | fixed concat | 0.5104 | 0.5079 |

Text is nearly uninformative because prompts repeat across emotion labels. Large
validation/test gaps also make the two-speaker pilot test unsuitable for selecting a
fusion design.

## Reproducibility checks

Training records a source/config snapshot, manifest hash, enabled cache contracts,
parameter counts, split/speaker/label/language/corpus counts, unique transcripts,
software versions, and diagnostic collapse flags. Evaluation refuses a checkpoint when
its model, labels, manifest, or cache contracts differ from the active configuration.

Paper-facing work still requires EmotionTalk cache extraction, frozen cross-corpus
tasks, and actual execution of the completed parameter-fair, episodic, ablation, and
repeated-seed paths.
