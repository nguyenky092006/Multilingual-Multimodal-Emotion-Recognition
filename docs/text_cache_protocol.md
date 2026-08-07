# Frozen Qwen3 text cache protocol

The text cache uses the SafeTensor revision
`Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
Original-language transcripts are tokenized without translation or a task instruction.
The final non-padding token is converted to float32, L2-normalized in float32, and stored as
a 1,024-dimensional vector. CUDA inference uses the model's bfloat16 weights.

CREMA-D contains only 12 exact prompted sentences. The extractor hashes the exact UTF-8
text, encodes each unique transcript once, and writes one SafeTensor per hash. A manifest-
specific JSONL index maps all 648 pilot samples (or all 4,900 full samples) to those shared
vectors. This preserves sample alignment without performing redundant encoder calls.

The audit file records prompt/emotion and prompt/split contingency counts, Cramer's V, and
mutual information without copying transcript text. The repeated prompts span all splits,
and `IEO` occurs more often for non-neutral emotions because intensity variants exist for
those classes. Text-only and trimodal results must therefore be reported with a prompt-
frequency-confounding warning; the text branch does not demonstrate emotional semantics on
CREMA-D by itself.

No network access is possible unless `--allow-download` is present:

```powershell
python scripts/cache_text_embeddings.py --dry-run --limit 16
python scripts/cache_text_embeddings.py --allow-download --limit 16
python scripts/cache_text_embeddings.py
```

The canary may download approximately 1.19 GB. The final command reuses local weights and
already verified unique transcript vectors to complete the pilot index.
