"""Frozen SigLIP frame embedding extraction with resumable SafeTensor caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor


VISUAL_CACHE_SCHEMA_VERSION = 1
SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class VisualCacheInput:
    """Minimal manifest view needed by the visual encoder."""

    sample_id: str
    video_path: str


@dataclass(frozen=True, slots=True)
class VisualClip:
    """Uniformly sampled RGB frames plus decoder and quality metadata."""

    frames: tuple[np.ndarray, ...]
    decoded_frame_count: int
    selected_indices: tuple[int, ...]
    width: int
    height: int
    fps: float | None
    duration_seconds: float | None
    codec: str
    mean_brightness: float
    mean_gradient_energy: float


@dataclass(slots=True)
class VisualCacheResult:
    output_dir: Path
    processed: int
    skipped: int
    selected: int
    embedding_dimension: int
    elapsed_seconds: float
    resolved_revision: str | None


FrameDecoder = Callable[[Path, int], VisualClip]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform_frame_indices(frame_count: int, frames_per_clip: int) -> tuple[int, ...]:
    """Return deterministic indices spanning the full decoded clip."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive")
    selected_count = min(frame_count, frames_per_clip)
    if selected_count == 1:
        return (frame_count // 2,)
    indices = np.rint(np.linspace(0, frame_count - 1, selected_count)).astype(np.int64)
    result = tuple(int(value) for value in indices)
    if len(set(result)) != selected_count:
        raise RuntimeError("uniform frame sampler produced duplicate indices")
    return result


def _frame_quality(frames: Sequence[np.ndarray]) -> tuple[float, float]:
    brightness: list[float] = []
    gradient_energy: list[float] = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("decoded frame must be uint8 RGB with shape [height,width,3]")
        gray = frame.astype(np.float32).mean(axis=2) / 255.0
        brightness.append(float(gray.mean()))
        vertical = np.diff(gray, axis=0)
        horizontal = np.diff(gray, axis=1)
        energy = float(np.mean(np.square(vertical)) + np.mean(np.square(horizontal)))
        gradient_energy.append(energy)
    return float(np.mean(brightness)), float(np.mean(gradient_energy))


def decode_uniform_frames(path: str | Path, frames_per_clip: int = 8) -> VisualClip:
    """Decode a video with PyAV and retain uniformly spaced full RGB frames."""

    source = Path(path)
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("install av==18.0.0 to decode visual clips") from exc
    try:
        container = av.open(str(source))
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        raise ValueError(f"cannot open video {source}: {exc}") from exc
    try:
        if not container.streams.video:
            raise ValueError(f"video contains no video stream: {source}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        fps = float(stream.average_rate) if stream.average_rate else None
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else None
        )
        codec = str(stream.codec_context.name or "unknown")
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        raise ValueError(f"cannot decode video {source}: {exc}") from exc
    finally:
        container.close()
    if not decoded:
        raise ValueError(f"video produced no decodable frames: {source}")
    indices = uniform_frame_indices(len(decoded), frames_per_clip)
    selected = tuple(np.ascontiguousarray(decoded[index]) for index in indices)
    first = selected[0]
    if any(frame.shape[:2] != first.shape[:2] for frame in selected):
        raise ValueError(f"selected frames change resolution within clip: {source}")
    brightness, gradient_energy = _frame_quality(selected)
    if duration is None and fps and fps > 0:
        duration = len(decoded) / fps
    return VisualClip(
        frames=selected,
        decoded_frame_count=len(decoded),
        selected_indices=indices,
        width=int(first.shape[1]),
        height=int(first.shape[0]),
        fps=fps,
        duration_seconds=duration,
        codec=codec,
        mean_brightness=brightness,
        mean_gradient_energy=gradient_energy,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _cache_path(output_dir: Path, sample_id: str) -> Path:
    if not SAFE_SAMPLE_ID.fullmatch(sample_id):
        raise ValueError(f"unsafe sample_id for cache filename: {sample_id!r}")
    return output_dir / "embeddings" / f"{sample_id}.safetensors"


def _load_components(
    identifier: str,
    revision: str | None,
    device: torch.device,
    inference_precision: str,
    allow_download: bool,
) -> tuple[Any, torch.nn.Module, str | None]:
    try:
        from transformers import AutoImageProcessor, SiglipVisionModel
    except ImportError as exc:
        raise RuntimeError("install the project encoder dependencies before extraction") from exc
    common = {"revision": revision, "local_files_only": not allow_download}
    dtype = torch.float16 if inference_precision == "float16" else torch.float32
    try:
        processor = AutoImageProcessor.from_pretrained(identifier, **common)
        model = SiglipVisionModel.from_pretrained(
            identifier,
            use_safetensors=True,
            dtype=dtype,
            **common,
        )
    except OSError as exc:
        if not allow_download:
            raise RuntimeError(
                "visual encoder weights are not available locally; rerun with --allow-download "
                "only after explicit model-download approval"
            ) from exc
        raise
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    return processor, model, str(resolved) if resolved else revision


def _pooled_frame_output(outputs: Any) -> Tensor:
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = getattr(outputs, "image_embeds", None)
    if pooled is None:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None or hidden.ndim != 3:
            raise RuntimeError("visual encoder returned no supported frame representation")
        pooled = hidden.mean(dim=1)
    if pooled.ndim != 2:
        raise RuntimeError(f"unexpected frame embedding shape: {tuple(pooled.shape)}")
    return pooled


def _existing_embedding(
    path: Path,
    expected_contract_hash: str,
    expected_dimension: int,
) -> tuple[bool, int]:
    if not path.is_file():
        return False, 0
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("contract_sha256") != expected_contract_hash:
                return False, 0
            embedding = handle.get_tensor("embedding")
        if (
            embedding.ndim != 1
            or embedding.numel() != expected_dimension
            or not torch.isfinite(embedding).all()
        ):
            return False, 0
        return True, int(embedding.numel())
    except (OSError, RuntimeError, ValueError):
        return False, 0


def cache_visual_embeddings(
    inputs: Sequence[VisualCacheInput],
    project_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    identifier: str,
    revision: str | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 2,
    frames_per_clip: int = 8,
    inference_precision: str = "float16",
    expected_embedding_dimension: int = 768,
    face_crop: bool = False,
    allow_download: bool = False,
    image_processor: Any | None = None,
    model: torch.nn.Module | None = None,
    resolved_revision: str | None = None,
    frame_decoder: FrameDecoder = decode_uniform_frames,
) -> VisualCacheResult:
    """Cache mean-pooled SigLIP full-frame vectors with strict resume checks."""

    if not inputs:
        raise ValueError("no visual samples selected")
    sample_ids = [item.sample_id for item in inputs]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("visual cache inputs contain duplicate sample_id values")
    if batch_size <= 0 or frames_per_clip <= 0:
        raise ValueError("batch_size and frames_per_clip must be positive")
    if expected_embedding_dimension <= 0:
        raise ValueError("expected_embedding_dimension must be positive")
    if inference_precision not in {"float16", "float32"}:
        raise ValueError("inference_precision must be float16 or float32")
    if face_crop:
        raise NotImplementedError("face crop is a later ablation; the baseline is full-frame")

    root = Path(project_root).resolve()
    manifest = Path(manifest_path)
    manifest = (root / manifest).resolve() if not manifest.is_absolute() else manifest.resolve()
    output = Path(output_dir)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"cache output directory must be inside project root: {output}") from exc
    output.mkdir(parents=True, exist_ok=True)
    (output / "embeddings").mkdir(exist_ok=True)

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    if inference_precision == "float16" and target_device.type != "cuda":
        raise ValueError("float16 inference is supported only on CUDA in this extractor")

    if image_processor is None or model is None:
        image_processor, model, loaded_revision = _load_components(
            identifier,
            revision,
            target_device,
            inference_precision,
            allow_download,
        )
        resolved_revision = loaded_revision
    else:
        model.requires_grad_(False)
        model.eval()
        model.to(target_device)
    assert image_processor is not None
    assert model is not None

    contract = {
        "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
        "modality": "visual",
        "model_identifier": identifier,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "frame_representation": "siglip_vision_pooler_output",
        "temporal_pooling": "mean",
        "frame_sampling": "uniform_full_clip_including_endpoints",
        "frames_per_clip": frames_per_clip,
        "face_crop": False,
        "decoder": "pyav",
        "inference_precision": inference_precision,
        "cache_dtype": "float32",
        "embedding_dimension": expected_embedding_dimension,
        "manifest_sha256": file_sha256(manifest),
    }
    contract_bytes = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    contract_hash = hashlib.sha256(contract_bytes).hexdigest()
    contract_path = output / "cache_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError(
                f"cache contract differs at {contract_path}; use a new output directory"
            )
    else:
        _atomic_json(contract_path, contract)

    pending: list[VisualCacheInput] = []
    skipped = 0
    embedding_dimension = 0
    for item in inputs:
        valid, dimension = _existing_embedding(
            _cache_path(output, item.sample_id),
            contract_hash,
            expected_embedding_dimension,
        )
        if valid:
            skipped += 1
            embedding_dimension = dimension
        else:
            pending.append(item)

    started = time.perf_counter()
    processed = 0
    for offset in range(0, len(pending), batch_size):
        batch_inputs = pending[offset : offset + batch_size]
        clips: list[VisualClip] = []
        source_paths: list[Path] = []
        flat_frames: list[np.ndarray] = []
        frame_counts: list[int] = []
        for item in batch_inputs:
            source = (root / item.video_path).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"video path is outside project root: {item.video_path}") from exc
            if not source.is_file():
                raise ValueError(f"video file does not exist: {source}")
            clip = frame_decoder(source, frames_per_clip)
            if not clip.frames:
                raise ValueError(f"video produced no selected frames: {source}")
            clips.append(clip)
            source_paths.append(source)
            flat_frames.extend(clip.frames)
            frame_counts.append(len(clip.frames))

        encoded = image_processor(images=flat_frames, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(target_device)
        autocast_enabled = inference_precision == "float16" and target_device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            outputs = model(pixel_values=pixel_values, return_dict=True)
            frame_embeddings = _pooled_frame_output(outputs)
        if frame_embeddings.shape[0] != len(flat_frames):
            raise RuntimeError("visual encoder output count does not match selected frame count")
        splits = torch.split(frame_embeddings, frame_counts, dim=0)
        pooled = torch.stack([part.mean(dim=0) for part in splits]).detach().float().cpu()
        if pooled.shape != (len(batch_inputs), expected_embedding_dimension):
            raise RuntimeError(f"unexpected pooled visual shape: {tuple(pooled.shape)}")
        if not torch.isfinite(pooled).all():
            raise RuntimeError("visual encoder produced non-finite embeddings")
        embedding_dimension = int(pooled.shape[1])

        from safetensors.torch import save_file

        for index, (item, clip, source) in enumerate(
            zip(batch_inputs, clips, source_paths, strict=True)
        ):
            target = _cache_path(output, item.sample_id)
            temporary = target.with_suffix(f".{os.getpid()}.safetensors.tmp")
            metadata = {
                "sample_id": item.sample_id,
                "source_video_path": item.video_path,
                "source_video_sha256": file_sha256(source),
                "decoder": "pyav",
                "codec": clip.codec,
                "decoded_frame_count": str(clip.decoded_frame_count),
                "valid_frame_count": str(len(clip.frames)),
                "selected_frame_indices": ",".join(map(str, clip.selected_indices)),
                "width": str(clip.width),
                "height": str(clip.height),
                "fps": "" if clip.fps is None else f"{clip.fps:.9g}",
                "duration_seconds": (
                    "" if clip.duration_seconds is None else f"{clip.duration_seconds:.9g}"
                ),
                "mean_brightness": f"{clip.mean_brightness:.9g}",
                "mean_gradient_energy": f"{clip.mean_gradient_energy:.9g}",
                "face_crop": "false",
                "contract_sha256": contract_hash,
            }
            save_file({"embedding": pooled[index].contiguous()}, temporary, metadata=metadata)
            os.replace(temporary, target)
            processed += 1

    index_rows: list[dict[str, Any]] = []
    from safetensors import safe_open

    for item in inputs:
        path = _cache_path(output, item.sample_id)
        valid, dimension = _existing_embedding(
            path,
            contract_hash,
            expected_embedding_dimension,
        )
        if not valid:
            raise RuntimeError(f"cache record failed post-write validation: {path}")
        embedding_dimension = dimension
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
        index_rows.append(
            {
                "sample_id": item.sample_id,
                "cache_path": path.relative_to(root).as_posix(),
                "embedding_dimension": dimension,
                **metadata,
            }
        )
    _atomic_jsonl(output / "index.jsonl", index_rows)

    elapsed = time.perf_counter() - started
    result = VisualCacheResult(
        output_dir=output,
        processed=processed,
        skipped=skipped,
        selected=len(inputs),
        embedding_dimension=embedding_dimension,
        elapsed_seconds=elapsed,
        resolved_revision=resolved_revision,
    )
    _atomic_json(
        output / "run_summary.json",
        {
            **asdict(result),
            "output_dir": output.relative_to(root).as_posix(),
            "device": str(target_device),
            "cuda_device": (
                torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None
            ),
            "torch_version": torch.__version__,
        },
    )
    return result
