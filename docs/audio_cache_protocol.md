# Frozen XLS-R audio cache protocol

The audio cache uses `facebook/wav2vec2-xls-r-300m` as a frozen multilingual waveform
encoder. The configuration pins the verified SafeTensor conversion revision
`e14e548dc8082338fa6e16f11ac95c532c0c6339` instead of following a mutable branch. Raw
CREMA-D PCM16 mono audio is already sampled at 16 kHz, so the extractor rejects rather than
silently resampling incompatible input.

Each variable-length hidden sequence is pooled with a feature-level padding mask. The
encoder stays in evaluation mode with gradients disabled. CUDA inference uses float16
autocast; the final one-dimensional vector is converted to float32 before storage.

One SafeTensor file is written atomically per sample. Its metadata records the source path
and hash, duration, RMS, peak, clipping fraction, model identifier/revision, manifest hash,
and cache-contract hash. Reruns verify each file and skip valid records. A changed manifest,
model, pooling rule, sample rate, or precision requires a new output directory.

No command downloads weights unless `--allow-download` is present. The intended sequence is:

```powershell
python scripts/cache_audio_embeddings.py --dry-run --limit 16
python scripts/cache_audio_embeddings.py --allow-download --limit 16
python scripts/cache_audio_embeddings.py
```

The second command is the explicitly approved canary and may download approximately 1.27 GB.
The third command reuses locally cached weights and the first 16 verified embeddings, then
completes the 648-sample pilot. Full-manifest extraction requires a separate output directory
and a separate approval after pilot inspection.
