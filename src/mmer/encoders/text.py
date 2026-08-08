"""Deduplicated Qwen3 text embedding extraction with SafeTensor caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor


TEXT_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class TextCacheInput:
    sample_id: str
    transcript: str
    emotion: str | None = None
    split: str | None = None
    sentence_code: str | None = None


@dataclass(slots=True)
class TextCacheResult:
    output_dir: Path
    index_path: Path
    selected_samples: int
    unique_transcripts: int
    processed_unique: int
    skipped_unique: int
    embedding_dimension: int
    elapsed_seconds: float
    resolved_revision: str | None


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def last_token_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the final non-padding token for either left- or right-padded batches."""

    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden state and attention mask shapes are incompatible")
    if not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError("text batch contains an empty token sequence")
    left_padded = bool(torch.all(attention_mask[:, -1] == 1).item())
    if left_padded:
        return hidden[:, -1]
    positions = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, positions]


def masked_mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean pool non-padding token states for smaller multilingual fallbacks."""

    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden state and attention mask shapes are incompatible")
    if not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError("text batch contains an empty token sequence")
    weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


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


def _cache_path(output_dir: Path, transcript_hash: str) -> Path:
    return output_dir / "embeddings" / f"{transcript_hash}.safetensors"


def _model_input(text: str, instruction: str | None) -> str:
    return text if not instruction else f"Instruct: {instruction}\nQuery: {text}"


def _load_components(
    identifier: str,
    revision: str | None,
    device: torch.device,
    inference_precision: str,
    allow_download: bool,
) -> tuple[Any, torch.nn.Module, str | None]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the project encoder dependencies before extraction") from exc
    common = {"revision": revision, "local_files_only": not allow_download}
    dtype = torch.bfloat16 if inference_precision == "bfloat16" else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(identifier, **common)
        model = AutoModel.from_pretrained(
            identifier,
            use_safetensors=True,
            dtype=dtype,
            **common,
        )
    except OSError as exc:
        if not allow_download:
            raise RuntimeError(
                "text encoder weights are not available locally; rerun with --allow-download "
                "only after explicit model-download approval"
            ) from exc
        raise
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    return tokenizer, model, str(resolved) if resolved else revision


def _existing_embedding(path: Path, contract_hash: str) -> tuple[bool, int, dict[str, str]]:
    if not path.is_file():
        return False, 0, {}
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            embedding = handle.get_tensor("embedding")
        valid = (
            metadata.get("contract_sha256") == contract_hash
            and embedding.ndim == 1
            and embedding.numel() > 0
            and bool(torch.isfinite(embedding).all())
        )
        return valid, int(embedding.numel()) if valid else 0, metadata if valid else {}
    except (OSError, RuntimeError, ValueError):
        return False, 0, {}


def prompt_label_audit(inputs: Sequence[TextCacheInput]) -> dict[str, Any]:
    """Quantify repeated-prompt label association without storing transcript text."""

    grouped: dict[str, list[TextCacheInput]] = defaultdict(list)
    for item in inputs:
        grouped[text_sha256(item.transcript)].append(item)
    labels = sorted({item.emotion for item in inputs if item.emotion})
    total = sum(1 for item in inputs if item.emotion)
    row_totals: dict[str, int] = {}
    column_totals = Counter(item.emotion for item in inputs if item.emotion)
    contingency: dict[str, Counter[str]] = {}
    prompts: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        emotion_counts = Counter(item.emotion for item in group if item.emotion)
        split_counts = Counter(item.split for item in group if item.split)
        row_totals[key] = sum(emotion_counts.values())
        contingency[key] = emotion_counts
        prompts.append(
            {
                "text_sha256": key,
                "sentence_codes": sorted({item.sentence_code for item in group if item.sentence_code}),
                "sample_count": len(group),
                "emotion_counts": dict(sorted(emotion_counts.items())),
                "split_counts": dict(sorted(split_counts.items())),
            }
        )
    chi_square = 0.0
    mutual_information_nats = 0.0
    if total and labels:
        for key, row_total in row_totals.items():
            for label in labels:
                observed = contingency[key][label]
                expected = row_total * column_totals[label] / total
                if expected > 0:
                    chi_square += (observed - expected) ** 2 / expected
                if observed > 0:
                    probability = observed / total
                    mutual_information_nats += probability * math.log(
                        observed * total / (row_total * column_totals[label])
                    )
    denominator_dimension = min(max(len(grouped) - 1, 0), max(len(labels) - 1, 0))
    cramers_v = math.sqrt(chi_square / (total * denominator_dimension)) if total and denominator_dimension else 0.0
    return {
        "sample_count": len(inputs),
        "unique_transcripts": len(grouped),
        "prompts_spanning_multiple_splits": sum(
            1 for group in grouped.values() if len({item.split for item in group if item.split}) > 1
        ),
        "cramers_v_prompt_vs_emotion": cramers_v,
        "mutual_information_bits": mutual_information_nats / math.log(2.0),
        "warning": (
            "Repeated prompted sentences span splits. Text-only performance can reflect "
            "prompt-frequency artifacts rather than emotional semantics."
        ),
        "prompts": prompts,
    }


def cache_text_embeddings(
    inputs: Sequence[TextCacheInput],
    project_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    identifier: str,
    revision: str | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 16,
    max_length: int = 128,
    inference_precision: str = "bfloat16",
    normalize_embeddings: bool = True,
    instruction: str | None = None,
    expected_embedding_dimension: int = 1024,
    pooling: str = "last_token",
    allow_download: bool = False,
    tokenizer: Any | None = None,
    model: torch.nn.Module | None = None,
    resolved_revision: str | None = None,
) -> TextCacheResult:
    """Encode exact unique transcripts once and map every sample through an index."""

    if not inputs:
        raise ValueError("no text samples selected")
    sample_ids = [item.sample_id for item in inputs]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("text cache inputs contain duplicate sample_id values")
    if any(not item.transcript.strip() for item in inputs):
        raise ValueError("text cache inputs contain an empty transcript")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    if inference_precision not in {"bfloat16", "float32"}:
        raise ValueError("inference_precision must be bfloat16 or float32")
    if pooling not in {"last_token", "masked_mean"}:
        raise ValueError("text pooling must be last_token or masked_mean")
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
    if inference_precision == "bfloat16":
        if target_device.type != "cuda":
            raise ValueError("bfloat16 inference is supported only on CUDA in this extractor")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16")

    if tokenizer is None or model is None:
        tokenizer, model, loaded_revision = _load_components(
            identifier, revision, target_device, inference_precision, allow_download
        )
        resolved_revision = loaded_revision
    else:
        tokenizer.padding_side = "left"
        model.requires_grad_(False)
        model.eval()
        model.to(target_device)
    assert tokenizer is not None
    assert model is not None

    contract = {
        "schema_version": TEXT_CACHE_SCHEMA_VERSION,
        "modality": "text",
        "model_identifier": identifier,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "pooling": pooling,
        "padding_side": "left",
        "max_length": max_length,
        "inference_precision": inference_precision,
        "cache_dtype": "float32",
        "normalize_embeddings": normalize_embeddings,
        "normalization_dtype": "float32",
        "instruction": instruction,
        "deduplication": "exact_utf8_sha256",
        "expected_embedding_dimension": expected_embedding_dimension,
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    contract_path = output / "cache_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError(f"cache contract differs at {contract_path}; use a new output directory")
    else:
        _atomic_json(contract_path, contract)

    unique: dict[str, str] = {}
    for item in inputs:
        unique.setdefault(text_sha256(item.transcript), item.transcript)
    pending: list[tuple[str, str]] = []
    skipped_unique = 0
    embedding_dimension = 0
    for transcript_hash, transcript in unique.items():
        valid, dimension, _ = _existing_embedding(
            _cache_path(output, transcript_hash), contract_hash
        )
        if valid:
            skipped_unique += 1
            embedding_dimension = dimension
        else:
            pending.append((transcript_hash, transcript))

    started = time.perf_counter()
    processed_unique = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        model_texts = [_model_input(text, instruction) for _, text in batch]
        encoded = tokenizer(
            model_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        attention_mask = encoded["attention_mask"]
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=False, return_dict=True)
            pooled = (
                last_token_pool(outputs.last_hidden_state, attention_mask)
                if pooling == "last_token"
                else masked_mean_pool(outputs.last_hidden_state, attention_mask)
            )
        pooled = pooled.detach().float().cpu()
        if normalize_embeddings:
            pooled = functional.normalize(pooled, p=2, dim=1)
        if pooled.ndim != 2 or pooled.shape[0] != len(batch):
            raise RuntimeError(f"unexpected pooled embedding shape: {tuple(pooled.shape)}")
        if pooled.shape[1] != expected_embedding_dimension:
            raise RuntimeError(
                f"expected {expected_embedding_dimension} text features, got {pooled.shape[1]}"
            )
        if not torch.isfinite(pooled).all():
            raise RuntimeError("text encoder produced non-finite embeddings")
        embedding_dimension = int(pooled.shape[1])
        from safetensors.torch import save_file

        token_counts = attention_mask.sum(dim=1).detach().cpu().tolist()
        for index, (transcript_hash, transcript) in enumerate(batch):
            target = _cache_path(output, transcript_hash)
            temporary = target.with_suffix(f".{os.getpid()}.safetensors.tmp")
            metadata = {
                "text_sha256": transcript_hash,
                "utf8_bytes": str(len(transcript.encode("utf-8"))),
                "unicode_characters": str(len(transcript)),
                "token_count": str(int(token_counts[index])),
                "contract_sha256": contract_hash,
            }
            save_file({"embedding": pooled[index].contiguous()}, temporary, metadata=metadata)
            os.replace(temporary, target)
            processed_unique += 1

    index_rows: list[dict[str, Any]] = []
    for item in inputs:
        transcript_hash = text_sha256(item.transcript)
        path = _cache_path(output, transcript_hash)
        valid, dimension, metadata = _existing_embedding(path, contract_hash)
        if not valid:
            raise RuntimeError(f"text cache record failed post-write validation: {path}")
        embedding_dimension = dimension
        index_rows.append(
            {
                "sample_id": item.sample_id,
                "text_sha256": transcript_hash,
                "cache_path": path.relative_to(root).as_posix(),
                "embedding_dimension": dimension,
                "token_count": int(metadata["token_count"]),
            }
        )
    manifest_stem = manifest.stem
    index_path = output / "indexes" / f"{manifest_stem}.jsonl"
    _atomic_jsonl(index_path, index_rows)
    audit = prompt_label_audit(inputs)
    audit.update({"manifest_sha256": _file_sha256(manifest), "manifest": manifest.relative_to(root).as_posix()})
    _atomic_json(output / "audits" / f"{manifest_stem}.json", audit)
    elapsed = time.perf_counter() - started
    result = TextCacheResult(
        output_dir=output,
        index_path=index_path,
        selected_samples=len(inputs),
        unique_transcripts=len(unique),
        processed_unique=processed_unique,
        skipped_unique=skipped_unique,
        embedding_dimension=embedding_dimension,
        elapsed_seconds=elapsed,
        resolved_revision=resolved_revision,
    )
    summary = asdict(result)
    summary["output_dir"] = output.relative_to(root).as_posix()
    summary["index_path"] = index_path.relative_to(root).as_posix()
    summary.update(
        {
            "manifest": manifest.relative_to(root).as_posix(),
            "manifest_sha256": _file_sha256(manifest),
            "device": str(target_device),
            "cuda_device": (
                torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None
            ),
            "torch_version": torch.__version__,
        }
    )
    _atomic_json(output / "summaries" / f"{manifest_stem}.json", summary)
    return result
