# Third-party notice

This first implementation is a clean-room implementation written for this repository.
No source code was copied or adapted from SER-Fuse or LaERC-S.

Resources consulted for research and interface decisions:

- SER-Fuse paper and official repository: https://github.com/nhattruongpham/SER-Fuse
- LaERC-S paper and official repository: https://github.com/bigcat-1/LaERC-S
- Qwen3 Embedding model card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Dataset sources listed in `docs/dataset_audit.md`

At the audit date, neither consulted research repository exposed a top-level license
that authorized code reuse. SER-Fuse contained notebooks; LaERC-S contained only a
minimal README. Consequently, this repository uses neither codebase as a dependency.
Dataset licenses are separate from code licenses and remain the responsibility of the
person obtaining each dataset.
