"""Frozen XLS-R audio embedding extraction with resumable SafeTensor caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor


CACHE_SCHEMA_VERSION = 1
SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class AudioCacheInput:
    """Minimal manifest view needed by the audio encoder."""

    sample_id: str
    audio_path: str


@dataclass(frozen=True, slots=True)
class AudioSignal:
    waveform: np.ndarray
    sample_rate: int
    source_sample_rate: int
    source_channels: int
    duration_seconds: float
    rms: float
    peak: float
    clipping_fraction: float


@dataclass(slots=True)
class AudioCacheResult:
    output_dir: Path
    processed: int
    skipped: int
    selected: int
    embedding_dimension: int
    elapsed_seconds: float
    resolved_revision: str | None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pcm16_mono(path: str | Path, expected_sample_rate: int = 16_000) -> AudioSignal:
    """Read PCM16 WAV, downmix channels, and linearly resample when required."""

    source = Path(path)
    try:
        with wave.open(str(source), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frames = stream.getnframes()
            compression = stream.getcomptype()
            payload = stream.readframes(frames)
    except (EOFError, OSError, wave.Error) as exc:
        raise ValueError(f"cannot decode WAV {source}: {exc}") from exc
    if channels <= 0:
        raise ValueError(f"audio has invalid channel count {channels}: {source}")
    if sample_width != 2 or compression != "NONE":
        raise ValueError(f"audio must be uncompressed PCM16: {source}")
    interleaved = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if interleaved.size % channels:
        raise ValueError(f"audio payload is not divisible by {channels} channels: {source}")
    waveform = interleaved.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"audio contains no samples: {source}")
    if sample_rate != expected_sample_rate:
        target_length = max(1, int(round(waveform.size * expected_sample_rate / sample_rate)))
        source_positions = np.arange(waveform.size, dtype=np.float64)
        target_positions = np.linspace(0.0, max(0, waveform.size - 1), target_length)
        waveform = np.interp(target_positions, source_positions, waveform).astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"audio contains no finite samples: {source}")
    square_mean = float(np.mean(np.square(waveform, dtype=np.float64)))
    return AudioSignal(
        waveform=waveform,
        sample_rate=expected_sample_rate,
        source_sample_rate=sample_rate,
        source_channels=channels,
        duration_seconds=float(waveform.size / expected_sample_rate),
        rms=math.sqrt(square_mean),
        peak=float(np.max(np.abs(waveform))),
        clipping_fraction=float(np.mean(np.abs(waveform) >= (32767.0 / 32768.0))),
    )


def masked_mean(hidden: Tensor, mask: Tensor | None) -> Tensor:
    """Mean-pool time steps while excluding padding."""

    if hidden.ndim != 3:
        raise ValueError(f"hidden state must have shape [batch,time,dim], got {tuple(hidden.shape)}")
    if mask is None:
        return hidden.mean(dim=1)
    if mask.shape != hidden.shape[:2]:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match {tuple(hidden.shape[:2])}")
    weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator


def masked_statistics(hidden: Tensor, mask: Tensor | None) -> Tensor:
    """Concatenate masked mean and standard deviation over time."""

    if hidden.ndim != 3:
        raise ValueError("hidden state must have shape [batch,time,dim]")
    valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device) if mask is None else mask.bool()
    if valid.shape != hidden.shape[:2] or (~valid).all(dim=1).any():
        raise ValueError("statistics mask is invalid or contains an empty sequence")
    weights = valid.to(hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    mean = (hidden * weights).sum(dim=1) / denominator
    variance = ((hidden - mean.unsqueeze(1)).square() * weights).sum(dim=1) / denominator
    return torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=-1)


def attentive_statistics(hidden: Tensor, mask: Tensor | None) -> Tensor:
    """Deterministic energy-attentive mean/std pooling for frozen representations."""

    if hidden.ndim != 3:
        raise ValueError("hidden state must have shape [batch,time,dim]")
    valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device) if mask is None else mask.bool()
    if valid.shape != hidden.shape[:2] or (~valid).all(dim=1).any():
        raise ValueError("attention mask is invalid or contains an empty sequence")
    logits = hidden.float().square().mean(dim=-1).to(hidden.dtype)
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=1).masked_fill(~valid, 0.0).unsqueeze(-1)
    mean = (hidden * weights).sum(dim=1)
    variance = ((hidden - mean.unsqueeze(1)).square() * weights).sum(dim=1)
    return torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=-1)


def pool_hidden(hidden: Tensor, mask: Tensor | None, pooling: str) -> Tensor:
    if pooling == "masked_mean":
        return masked_mean(hidden, mask)
    if pooling == "statistics":
        return masked_statistics(hidden, mask)
    if pooling == "attentive_statistics":
        return attentive_statistics(hidden, mask)
    raise ValueError(f"unsupported audio pooling: {pooling}")


def _chunk_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    max_duration_seconds: float,
    duration_policy: str,
    chunk_overlap_seconds: float,
) -> list[np.ndarray]:
    maximum = int(round(max_duration_seconds * sample_rate))
    if maximum <= 0:
        raise ValueError("max_duration_seconds must be positive")
    if waveform.size <= maximum:
        return [waveform]
    if duration_policy == "reject":
        raise ValueError("audio exceeds max_duration_seconds under reject policy")
    if duration_policy == "truncate":
        return [waveform[:maximum]]
    overlap = int(round(chunk_overlap_seconds * sample_rate))
    if duration_policy != "chunk" or overlap < 0 or overlap >= maximum:
        raise ValueError("chunk policy requires 0 <= overlap < max duration")
    step = maximum - overlap
    chunks = [waveform[start : start + maximum] for start in range(0, waveform.size, step)]
    return [chunk for chunk in chunks if chunk.size > 0]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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
    allow_download: bool,
) -> tuple[Any, torch.nn.Module, str | None]:
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise RuntimeError("install the project encoder dependencies before extraction") from exc
    common = {
        "revision": revision,
        "local_files_only": not allow_download,
    }
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(identifier, **common)
        model = AutoModel.from_pretrained(identifier, use_safetensors=True, **common)
    except OSError as exc:
        if not allow_download:
            raise RuntimeError(
                "encoder weights are not available locally; rerun with --allow-download only "
                "after explicit model-download approval"
            ) from exc
        raise
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    return feature_extractor, model, str(resolved) if resolved else revision


def _feature_mask(model: torch.nn.Module, hidden: Tensor, attention_mask: Tensor | None) -> Tensor | None:
    if attention_mask is None:
        return None
    method = getattr(model, "_get_feature_vector_attention_mask", None)
    if method is None:
        if attention_mask.shape[1] != hidden.shape[1]:
            raise RuntimeError("encoder cannot convert waveform padding mask to feature padding mask")
        return attention_mask.bool()
    return method(hidden.shape[1], attention_mask).bool()


def _existing_embedding(path: Path, expected_contract_hash: str) -> tuple[bool, int]:
    if not path.is_file():
        return False, 0
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("contract_sha256") != expected_contract_hash:
                return False, 0
            embedding = handle.get_tensor("embedding")
        if embedding.ndim != 1 or embedding.numel() == 0 or not torch.isfinite(embedding).all():
            return False, 0
        return True, int(embedding.numel())
    except (OSError, RuntimeError, ValueError):
        return False, 0


def cache_audio_embeddings(
    inputs: Sequence[AudioCacheInput],
    project_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    identifier: str,
    revision: str | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 2,
    sample_rate: int = 16_000,
    max_duration_seconds: float = 12.0,
    duration_policy: str = "reject",
    chunk_overlap_seconds: float = 0.0,
    pooling: str = "masked_mean",
    hidden_layer: int = -1,
    inference_precision: str = "float16",
    allow_download: bool = False,
    feature_extractor: Any | None = None,
    model: torch.nn.Module | None = None,
    resolved_revision: str | None = None,
) -> AudioCacheResult:
    """Extract masked-mean XLS-R vectors, resuming verified per-sample cache files."""

    if not inputs:
        raise ValueError("no audio samples selected")
    sample_ids = [item.sample_id for item in inputs]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("audio cache inputs contain duplicate sample_id values")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if inference_precision not in {"float16", "float32"}:
        raise ValueError("inference_precision must be float16 or float32")
    if duration_policy not in {"reject", "truncate", "chunk"}:
        raise ValueError("duration_policy must be reject, truncate, or chunk")
    if chunk_overlap_seconds < 0 or chunk_overlap_seconds >= max_duration_seconds:
        raise ValueError("chunk_overlap_seconds must be in [0, max_duration_seconds)")
    if pooling not in {"masked_mean", "statistics", "attentive_statistics"}:
        raise ValueError("unsupported audio pooling")
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

    manifest_hash = file_sha256(manifest)
    if feature_extractor is None or model is None:
        feature_extractor, model, loaded_revision = _load_components(
            identifier, revision, target_device, allow_download
        )
        resolved_revision = loaded_revision
    else:
        model.requires_grad_(False)
        model.eval()
        model.to(target_device)
    assert model is not None
    assert feature_extractor is not None

    contract = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "modality": "audio",
        "model_identifier": identifier,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "pooling": pooling,
        "sample_rate": sample_rate,
        "max_duration_seconds": max_duration_seconds,
        "inference_precision": inference_precision,
        "cache_dtype": "float32",
        "manifest_sha256": manifest_hash,
    }
    if hidden_layer != -1:
        contract["hidden_layer"] = int(hidden_layer)
    if duration_policy != "reject":
        contract["duration_policy"] = duration_policy
        if duration_policy == "chunk":
            contract["chunk_overlap_seconds"] = float(chunk_overlap_seconds)
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

    pending: list[AudioCacheInput] = []
    skipped = 0
    embedding_dimension = 0
    for item in inputs:
        valid, dimension = _existing_embedding(_cache_path(output, item.sample_id), contract_hash)
        if valid:
            skipped += 1
            embedding_dimension = dimension
        else:
            pending.append(item)

    started = time.perf_counter()
    processed = 0
    for offset in range(0, len(pending), batch_size):
        batch_inputs = pending[offset : offset + batch_size]
        signals: list[AudioSignal] = []
        source_paths: list[Path] = []
        segment_waveforms: list[np.ndarray] = []
        segment_owners: list[int] = []
        segment_counts: list[int] = []
        for item in batch_inputs:
            source = (root / item.audio_path).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"audio path is outside project root: {item.audio_path}") from exc
            signal = read_pcm16_mono(source, sample_rate)
            try:
                segments = _chunk_waveform(
                    signal.waveform,
                    sample_rate,
                    max_duration_seconds,
                    duration_policy,
                    chunk_overlap_seconds,
                )
            except ValueError as exc:
                raise ValueError(
                    f"{item.sample_id} duration {signal.duration_seconds:.3f}s exceeds "
                    f"{max_duration_seconds:.3f}s: {exc}"
                ) from exc
            signals.append(signal)
            source_paths.append(source)
            owner = len(signals) - 1
            segment_waveforms.extend(segments)
            segment_owners.extend([owner] * len(segments))
            segment_counts.append(len(segments))
        encoded = feature_extractor(
            segment_waveforms,
            sampling_rate=sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = encoded["input_values"].to(target_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)
        autocast_enabled = inference_precision == "float16" and target_device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=hidden_layer != -1,
                return_dict=True,
            )
            if hidden_layer == -1:
                hidden = outputs.last_hidden_state
            else:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError("audio encoder did not return requested hidden states")
                try:
                    hidden = hidden_states[hidden_layer]
                except IndexError as exc:
                    raise ValueError(f"hidden_layer {hidden_layer} is out of range") from exc
            segment_pooled = pool_hidden(
                hidden, _feature_mask(model, hidden, attention_mask), pooling
            )
        sample_rows = [
            segment_pooled[
                torch.tensor(
                    [index for index, value in enumerate(segment_owners) if value == owner],
                    device=segment_pooled.device,
                )
            ].mean(dim=0)
            for owner in range(len(signals))
        ]
        pooled = torch.stack(sample_rows).detach().float().cpu()
        if pooled.ndim != 2 or pooled.shape[0] != len(batch_inputs):
            raise RuntimeError(f"unexpected pooled embedding shape: {tuple(pooled.shape)}")
        if not torch.isfinite(pooled).all():
            raise RuntimeError("encoder produced non-finite embeddings")
        embedding_dimension = int(pooled.shape[1])
        from safetensors.torch import save_file

        for index, (item, signal, source) in enumerate(zip(batch_inputs, signals, source_paths, strict=True)):
            target = _cache_path(output, item.sample_id)
            temporary = target.with_suffix(f".{os.getpid()}.safetensors.tmp")
            metadata = {
                "sample_id": item.sample_id,
                "source_audio_path": item.audio_path,
                "source_audio_sha256": file_sha256(source),
                "source_sample_rate": str(signal.source_sample_rate),
                "source_channels": str(signal.source_channels),
                "chunk_count": str(segment_counts[index]),
                "duration_seconds": f"{signal.duration_seconds:.9f}",
                "rms": f"{signal.rms:.9g}",
                "peak": f"{signal.peak:.9g}",
                "clipping_fraction": f"{signal.clipping_fraction:.9g}",
                "contract_sha256": contract_hash,
            }
            save_file({"embedding": pooled[index].contiguous()}, temporary, metadata=metadata)
            os.replace(temporary, target)
            processed += 1

    index_rows: list[dict[str, Any]] = []
    from safetensors import safe_open

    for item in inputs:
        path = _cache_path(output, item.sample_id)
        valid, dimension = _existing_embedding(path, contract_hash)
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
    result = AudioCacheResult(
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
