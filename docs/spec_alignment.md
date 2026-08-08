# Specification alignment

The repository has two related but different research tracks. Keeping them separate
prevents the labels `P1`, `P2`, and `P3` from silently changing meaning.

## Paper audio-text track

The two Word documents define an English-Mandarin audio-and-transcript study:

- P1: hierarchical language/corpus-aware adapters plus a reliability gate;
- P2: P1 plus Prototypical Networks;
- P3: P2 plus a supervised contrastive objective.

This track is not implemented end to end yet because no approved multilingual dataset,
paper manifest, or paper cache contract is present in the repository. ProtoNet is also
explicitly deferred by `AGENTS.md`. CREMA-D can only be an English external diagnostic;
it cannot establish the multilingual claim.

## Reusable framework track

The later implementation prompt expands the reusable software contract to independent
audio, text, and visual streams:

- F1: projections and hierarchical adapters with masked concatenation;
- F2: F1 plus reliability-aware fusion;
- F3: F2 plus future Prototypical Networks.

The CREMA-D supervised reliability experiment implements F2. Visual is therefore an
intentional framework capability, not evidence that the paper documents specified a
visual branch. The descriptive configuration is
`configs/experiment/cremad_pilot_supervised_reliability.yaml`; the old
`cremad_pilot_p2.yaml` name is retained only as a deprecated alias.

## Current evidence boundary

All current CREMA-D runs are pilot or engineering diagnostics. They are single-corpus,
English-only, and use twelve fixed prompted transcripts. They must remain marked
`paper_ready: false`. Paper-facing claims require a multilingual protocol, repeated
seeds, parameter-fair baselines, and the later episodic implementation.
