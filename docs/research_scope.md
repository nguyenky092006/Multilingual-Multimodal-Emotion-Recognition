# Research scope

## Task

This repository studies **multilingual multimodal emotion recognition (MMER)**.
One example is one independent utterance or video clip with an emotion label and
zero or more audio, original-language text, and visual streams. Conversation history,
previous turns, speaker-history reasoning, dialogue graphs, and federated learning are
outside the first implementation.

The initial common label space is `angry`, `happy`, `neutral`, and `sad`. A source
dataset is usable only after its labels have been mapped explicitly. In particular,
`excited` is not silently merged into `happy`.

## Research question and hypotheses

The central question is whether frozen encoders plus lightweight modules can separate
shared emotion knowledge from modality-, language-, and corpus-specific residuals.
The planned comparisons test negative transfer from fully shared training, routed
language/corpus adapters, robustness of reliability-aware fusion, and eventually
few-shot target-corpus adaptation with Prototypical Networks.

This iteration tests only architecture and software correctness on synthetic cached
embeddings. It does not make a scientific performance claim.

## Evidence boundaries

- English and Mandarin are supported route values, but a mixed batch is not evidence
  of language generalization.
- Country, region, and accent are nullable metadata. No model route or result is named
  for them until verified labels exist.
- Missing video means `visual_available=false`; an audio or text vector is never copied
  into the visual branch.
- Encoders remain outside the training process after cache creation. No large model or
  dataset is downloaded in this iteration.
- Synthetic metrics are software diagnostics and must be labelled `synthetic`.

## Initial protocol recommendation

No verified, openly redistributable, speaker-balanced English–Mandarin trimodal corpus
was found that removes language–corpus confounding. The honest initial real-data design
is therefore cross-corpus:

1. use a public/licensed English trimodal corpus such as CREMA-D;
2. use a separately licensed Mandarin trimodal source such as CH-SIMS v2/EmoS static
   annotations or CHEAVD only after its exact access terms are confirmed;
3. split by speaker and original source video before producing clips;
4. report within-corpus, leave-one-corpus-out, and per-language scores separately;
5. state that language and corpus are entangled and do not interpret a difference as a
   purely linguistic effect.

ESD can be a bilingual audio-plus-text control because its English and Mandarin subsets
share a protocol. It is not a trimodal dataset and cannot supply visual evidence.

