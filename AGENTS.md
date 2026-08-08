# Repository guidance

- The task is utterance/clip-level multilingual multimodal emotion recognition, not conversational ERC.
- Keep audio, text, and visual inputs independent and represent missing modalities explicitly.
- Never fabricate paths, transcripts, video features, demographic attributes, or experimental results.
- Do not download datasets or pretrained weights without explicit user approval.
- Treat `references/source/`, datasets, embedding caches, checkpoints, and credentials as private/untracked.
- Preserve the four-label mapping in `configs/data/labels.yaml`; incompatible source labels require an explicit mapping decision.
- Frozen encoders and cache extraction are separate from adapter/fusion training.
- Tests and the CPU smoke run must stay offline and deterministic.
- ProtoNet and supervised contrastive episodic learning are available only after the
  verified cached-embedding pipeline. Federated learning remains outside this project.
