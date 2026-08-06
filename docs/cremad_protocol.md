# CREMA-D preparation protocol

This protocol records the decisions approved for the first real-data experiment. It is
deliberately dataset-specific; it does not redefine the label policy of other corpora.

## Labels and text

- The target is the intended emotion encoded in the official filename.
- `ANG`, `HAP`, `NEU`, and `SAD` map to `angry`, `happy`, `neutral`, and `sad`.
- `DIS` and `FEA` are excluded. They are never merged into another class.
- Text is the official prompted sentence selected by its three-letter sentence code.
- `processedResults/summaryTable.csv` is retained for later annotation-agreement analysis,
  but its perceptual votes are not used as the training target in this protocol.

The expected full four-class manifest contains 4,900 clips when the official clone is
complete: 1,271 angry, 1,271 happy, 1,087 neutral, and 1,271 sad.

## Speaker-exclusive splits

Actors are grouped by `ActorID` before samples are assigned. With seed 17, the full split
uses 64 train, 14 validation, and 13 test actors. Assignment is stratified by the recorded
`Sex` field, yielding 34/30, 7/7, and 7/6 male/female actors respectively for the official
48-male/43-female demographics.

The pilot is selected inside those fixed full splits and uses 8 train, 2 validation, and
2 test actors, balanced by recorded sex. It changes only dataset size; labels, modalities,
and model settings remain aligned with the full experiment. Pilot metrics are debugging
signals and must not be reported as final results.

Age, sex, race, and ethnicity are written only to the audit report through the list of
available demographic fields. They are not manifest model inputs and are not used as
features.

## Build and validate

From the repository root:

```powershell
python scripts/build_cremad_manifest.py
python scripts/validate_manifest.py data/manifests/cremad_full.jsonl --labels configs/data/labels.yaml --data-root .
python scripts/validate_manifest.py data/manifests/cremad_pilot.jsonl --labels configs/data/labels.yaml --data-root .
```

The builder writes:

- `data/manifests/cremad_full.jsonl`
- `data/manifests/cremad_pilot.jsonl`
- `data/manifests/cremad_speaker_splits.json`
- `data/manifests/cremad_build_report.json`

All generated files remain under the ignored `data/` directory. Repeated official prompt
sentences across splits produce expected duplicate-transcript warnings; speaker or media
leakage and corrupt files remain fatal. The lightweight video check verifies the container
header. Encoder extraction should additionally decode/inspect frames and record any media
that fails at that stage.
