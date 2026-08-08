# CREMA-D full audio-text baseline protocol

## Scope

These configurations use the approved 64/14/13 speaker split and all 4,900 retained
ANG/HAP/NEU/SAD clips. They compare frozen XLS-R audio, frozen Qwen3 text, and fixed
audio-text concatenation under the same classifier dimensions and optimization policy.

CREMA-D is English-only and has twelve repeated official prompts. These runs are
full-corpus engineering baselines, not evidence for multilingual transfer or the paper's
few-shot claims. Every output remains `paper_ready: false`.

## Fixed seed-17 configurations

```powershell
python scripts/train.py --config configs/experiment/cremad_full_baseline_audio.yaml
python scripts/evaluate.py --config configs/experiment/cremad_full_baseline_audio.yaml

python scripts/train.py --config configs/experiment/cremad_full_baseline_text.yaml
python scripts/evaluate.py --config configs/experiment/cremad_full_baseline_text.yaml

python scripts/train.py --config configs/experiment/cremad_full_baseline_audio_text_concat.yaml
python scripts/evaluate.py --config configs/experiment/cremad_full_baseline_audio_text_concat.yaml
```

All three runs use seed 17, class weights, batch size 64, learning rate 0.001, no
modality dropout, no adapters, no routing, and validation-UAR checkpoint selection.
Audio-only and text-only have identical trainable architecture sizes because both input
vectors are 1,024-D. Audio-text has two projection branches and a wider concatenation
layer, so its parameter count must be reported rather than called parameter-matched.

## Interpretation gate

Compare test UAR, macro-F1, accuracy, per-class recall, and validation-test gaps. Do not
select a method from test performance after each run. After the seed-17 pipeline is
verified, freeze the configurations and repeat seeds 23 and 42 before calculating means,
standard deviations, or confidence intervals.
