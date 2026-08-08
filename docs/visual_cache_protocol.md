# Frozen visual cache protocol

## Scope

The initial visual baseline uses `google/siglip-base-patch16-224` as a frozen frame
encoder. Each clip is decoded with PyAV, eight RGB full frames are sampled uniformly
across the decoded clip, the official Hugging Face image preprocessing is applied, and
the SigLIP vision pooler outputs are averaged across valid frames. The resulting
768-dimensional vector is saved as float32.

This is a generic image-language representation, not a facial-expression-specific
encoder. The baseline remains full-frame mean pooling. Separate contracts implement an
optional OpenCV Haar largest-face crop with full-frame fallback and frame-level caches
for trainable temporal attention. No missing or corrupt video is replaced with fabricated
features.

## Reproducibility contract

- Model: `google/siglip-base-patch16-224`
- Revision: `559dc4dd2ab64df71cea758c5204a43113812dae`
- License recorded by the model repository: Apache-2.0
- Decoder: PyAV 18.x with its bundled FFmpeg libraries
- Sampling: eight uniformly spaced frames including both clip endpoints
- Image preprocessing: the checkpoint's official `AutoImageProcessor`
- Frame representation: `SiglipVisionModel` pooler output
- Temporal aggregation: arithmetic mean over valid sampled frames
- Inference: float16 on CUDA; cache: float32
- Face crop: disabled

The baseline values above are not silently changed. Temporal attention stores the valid
`frame_embeddings` matrix and learns attention inside the classifier. Face-crop mode
records detected/fallback counts and requires the optional `vision` dependency.

The cache contract includes the manifest hash, requested and resolved model revisions,
sampling policy, frame count, pooling, precision, face-crop flag, and embedding
dimension. Reusing an output directory with a different contract is rejected.

Every sample SafeTensor stores the source-video SHA-256, codec, decoded/selected frame
counts, selected indices, resolution, FPS/duration when available, mean brightness,
gradient energy, and contract hash. Writes are atomic and existing records are resumed
only after their tensor and contract are validated.

## Safe execution order

Inspect configuration without decoding video or loading a model:

```powershell
python scripts/cache_visual_embeddings.py --dry-run --limit 16
```

Check real FLV decoding and frame sampling without loading or downloading SigLIP:

```powershell
python scripts/cache_visual_embeddings.py --decode-check --limit 16
```

Only after explicit approval for the approximately 813 MB SafeTensor checkpoint, run a
16-clip canary:

```powershell
python scripts/cache_visual_embeddings.py --allow-download --limit 16
```

If the canary passes, resume the complete 648-clip pilot from local model files:

```powershell
python scripts/cache_visual_embeddings.py
```

Repeating either extraction command is safe: verified vectors are skipped. Data, model
weights, and generated caches remain ignored by Git.

## Verified CREMA-D pilot

The approved pilot extraction completed on 2026-08-07 with the pinned revision above:

- 648 manifest records, 648 SafeTensor files, and 648 index rows;
- 16 canary records resumed and 632 additional records processed;
- 768-dimensional float32 vectors, all finite, nonzero, and byte-distinct;
- exactly eight valid frames per clip at the source resolution 480×360;
- VP6F and H.264 source codecs decoded successfully;
- one consistent cache-contract hash across every record;
- visual index order exactly matched the pilot manifest;
- audio, text, and visual index sample-ID sets all matched the same 648 records;
- extraction time for the remaining 632 clips: 349.93 seconds on the local RTX 4060
  Laptop GPU;
- generated visual cache size: approximately 2.92 MB, excluding model weights.

The model loader reports text-branch keys as `UNEXPECTED` because
`SiglipVisionModel` intentionally loads only the vision submodel from the multimodal
SigLIP checkpoint. These keys do not indicate missing vision weights.

## Interpretation limits

- Full frames retain background, illumination, clothing, and identity cues. Results may
  therefore contain corpus- or speaker-specific shortcuts.
- Uniform sampling is a reproducible low-compute baseline, not an optimal temporal model.
- Mean brightness and gradient energy are quality diagnostics, not validated face-quality
  estimates.
- The registered controlled study compares full frame/face crop and mean/temporal
  attention using separate immutable caches. A facial-expression-specific encoder is
  still an optional future representation comparison, not a core requirement.
