# EmotionTalk audio-text protocol

## Scope and evidence boundary

EmotionTalk is added as the Mandarin audio-text corpus for the independent-utterance
SER path. This converter uses only the extracted `Audio` archive. It does not require
`Multimodal.tar`, does not claim a visual result, and does not treat the dataset as a
conversational ERC benchmark.

Before publication, verify the current upstream dataset card and comply with its
CC BY-NC-SA 4.0 terms and gated-access agreement. Raw data, archives, manifests,
embedding caches, and model outputs remain untracked.

## Target and transcript policy

- Target: `Audio/json/**/emotion_result`.
- Transcript: the matching `Audio/json/**/content`, preserved in original Chinese.
- Included labels: `angry`, `happy`, `neutral`, `sad`.
- Excluded without merging: `disgusted`, `fearful`, `surprised`.
- `Text.tar` emotion labels are not mixed into the target. They are modality-specific
  annotations and their disagreement with the Audio label is not a data error.
- Annotator votes under `data` are audited; `emotion_result` remains the official target.

## Leakage-safe split

The converter builds a graph whose vertices are speaker IDs and whose edges connect
speakers occurring in the same dialogue. Every connected component is assigned wholly
to one split. It then chooses the deterministic assignment that best matches the
70/15/15 sample targets while retaining all four labels in every split. Entire
top-level source groups (`G000xx`) must also remain within one split or the build stops.

For the currently approved EmotionTalk release, this yields approximately 71.4% train,
13.4% validation, and 15.3% test by four-class utterance count. The generated split JSON
records exact speakers, components, sources, and class support.

The pilot is a deterministic class-balanced subset of the already leakage-safe full
split: 64 samples per class for train and 16 per class for validation/test. It is an
engineering canary, not paper-ready evidence.

The verified four-class release contains 3,820 angry, 2,105 happy, 9,378 neutral, and
1,110 sad utterances. The build therefore emits a class-imbalance warning; UAR and
macro-F1 remain primary metrics, and episodic sampling must stay class balanced. It also
reports 35 exact-transcript groups spanning splits. These are independent recordings
from separated speakers/sources, but the lexical-overlap limitation must be disclosed.

Seven pilot clips and 280 full-manifest clips exceed 12 seconds; the longest verified
clip is 24.900 seconds. Audio extraction therefore uses
`configs/encoder/emotiontalk_xlsr_chunk12s.yaml`, which preserves every clip as
non-overlapping chunks and duration-weights their pooled embeddings. The default
CREMA-D rejection contract must not be reused for EmotionTalk.

## Pilot experiments

The verified pilot cache supports four diagnostic configurations:

- `emotiontalk_pilot_baseline_audio.yaml`
- `emotiontalk_pilot_baseline_text.yaml`
- `emotiontalk_pilot_baseline_audio_text_concat.yaml`
- `emotiontalk_pilot_p1_reliability.yaml`

All four are Mandarin-only, speaker/source-disjoint, and explicitly marked diagnostic.
They do not require visual data. The first three isolate audio, text, and simple
concatenation; P1 then adds audio/text adapters, routing, and reliability fusion. The
official Audio transcript receives text quality 1.0 under the validated metadata policy.

## Build and audit

From the repository root after extracting `Audio.tar`:

```powershell
python scripts/build_emotiontalk_manifest.py
```

For a quicker schema/split rebuild that still checks presence and size of every WAV but
does not decode every WAV header:

```powershell
python scripts/build_emotiontalk_manifest.py --skip-file-checks
```

Outputs are:

- `data/manifests/emotiontalk_full.jsonl`
- `data/manifests/emotiontalk_pilot.jsonl`
- `data/manifests/emotiontalk_speaker_splits.json`
- `data/manifests/emotiontalk_build_report.json`

The normal build verifies JSON/WAV one-to-one paths, UTF-8 transcripts, positive
durations, WAV headers, JSON-vs-WAV duration agreement, label policy, speaker/source
separation, and the existing manifest contract. The local release audit found 18,612
stereo 44.1 kHz files, 518 mono 44.1 kHz files, and 120 mono 16 kHz files. The audio
cache pipeline handles both mono/stereo input and resamples before XLS-R inference.
