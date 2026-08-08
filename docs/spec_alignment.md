# Specification alignment

The repository has two related but different research tracks. Keeping them separate
prevents the labels `P1`, `P2`, and `P3` from silently changing meaning.

## Paper audio-text track

The two Word documents define an English-Mandarin audio-and-transcript study:

- P1: hierarchical language/corpus-aware adapters plus a reliability gate;
- P2: P1 plus Prototypical Networks;
- P3: P2 plus a supervised contrastive objective.

P1/P2/P3 are executable on verified cached embeddings. CREMA-D caches and engineering
configs are complete, and the approved EmotionTalk Mandarin audio-text converter is now
present. EmotionTalk caches and cross-corpus experiment configs are not yet complete,
so current runs still cannot establish multilingual or unseen-corpus claims.

## Reusable framework track

The later implementation prompt expands the reusable software contract to independent
audio, text, and visual streams:

- F1: projections and hierarchical adapters with masked concatenation;
- F2: F1 plus reliability-aware fusion;
- F3: F2 plus Prototypical Networks and optional supervised contrastive learning.

The CREMA-D supervised reliability experiment implements F2 and the generic episodic
runner implements F3 for any enabled modality subset. Visual is therefore an
intentional framework capability, not evidence that the paper documents specified a
visual branch. The descriptive configuration is
`configs/experiment/cremad_pilot_supervised_reliability.yaml`; the old
`cremad_pilot_p2.yaml` name is retained only as a deprecated alias.

## Current evidence boundary

All current CREMA-D runs are pilot or engineering diagnostics. They are single-corpus,
English-only, and use twelve fixed prompted transcripts. They must remain marked
`paper_ready: false`. Paper-facing claims still require EmotionTalk cache extraction,
a frozen cross-corpus protocol, repeated seeds, parameter-fair baselines, and explicit
language-corpus confounding analysis. The CREMA-D P1/P2/P3 files are deliberately
marked engineering diagnostics.
