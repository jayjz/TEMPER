"""Frozen EXP-0001 B2 bidirectional encoder baseline.

Scientific choices live in docs/exec-plans/active/EXP-0001-B2-FREEZE.md and are
not tunable here.
"""

from __future__ import annotations

import math
import os
import platform
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from temper.evaluation import BaselineMetrics, compute_baseline_metrics

B2_MODEL_ID = "google-bert/bert-base-uncased"
B2_MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
B2_TOKENIZER_ID = B2_MODEL_ID
B2_TOKENIZER_REVISION = B2_MODEL_REVISION
B2_NUM_LABELS = 150
B2_MAX_LENGTH = 128
B2_BATCH_SIZE = 16
B2_LEARNING_RATE = 2e-5
B2_WEIGHT_DECAY = 0.01
B2_EPOCHS = 3
B2_ADAMW_BETAS: tuple[float, float] = (0.9, 0.999)
B2_ADAMW_EPS = 1e-8
B2_WARMUP_RATIO = 0.1
B2_MAX_GRAD_NORM = 1.0
B2_SEEDS: tuple[int, ...] = (13, 21, 37)
B2_PARTITION = "validation"
B2_N_VALIDATION_EXAMPLES = 1500
B2_HIDDEN_SIZE = 768
B2_NUM_HIDDEN_LAYERS = 12
B2_NUM_ATTENTION_HEADS = 12
B2_VOCAB_SIZE = 30522
B2_MAX_POSITION_EMBEDDINGS = 512


@dataclass(frozen=True)
class B2RunArtifacts:
    """Seed-specific output names; paths stay under the caller-chosen output root."""

    run_id: str
    result_name: str
    manifest_name: str


def frozen_b2_hyperparameters() -> dict[str, object]:
    """Serialize the frozen training configuration for manifests."""
    return {
        "model_id": B2_MODEL_ID,
        "model_revision": B2_MODEL_REVISION,
        "tokenizer_id": B2_TOKENIZER_ID,
        "tokenizer_revision": B2_TOKENIZER_REVISION,
        "architecture": "BertForSequenceClassification",
        "num_labels": B2_NUM_LABELS,
        "max_length": B2_MAX_LENGTH,
        "batch_size": B2_BATCH_SIZE,
        "gradient_accumulation": None,
        "learning_rate": B2_LEARNING_RATE,
        "weight_decay": B2_WEIGHT_DECAY,
        "epochs": B2_EPOCHS,
        "optimizer": "torch.optim.AdamW",
        "betas": list(B2_ADAMW_BETAS),
        "eps": B2_ADAMW_EPS,
        "scheduler": "linear",
        "warmup_ratio": B2_WARMUP_RATIO,
        "max_grad_norm": B2_MAX_GRAD_NORM,
        "checkpoint_rule": "final-epoch-only",
        "early_stopping": None,
        "padding": "max_length",
        "truncation": True,
        "allowed_seeds": list(B2_SEEDS),
        "partition": B2_PARTITION,
    }


def require_allowed_b2_seed(seed: int) -> int:
    if seed not in B2_SEEDS:
        raise ValueError(f"B2 seed must be one of {B2_SEEDS}, found {seed}")
    return seed


def require_b2_validation_partition(partition: str) -> str:
    if partition != B2_PARTITION:
        raise ValueError(
            "B2 may only evaluate the frozen validation partition; "
            f"rejected partition {partition!r}"
        )
    return partition


def b2_run_artifacts(seed: int) -> B2RunArtifacts:
    seed = require_allowed_b2_seed(seed)
    run_id = f"EXP-0001-B2-{B2_PARTITION}-seed-{seed}"
    return B2RunArtifacts(
        run_id=run_id,
        result_name=f"{run_id}.npz",
        manifest_name=f"{run_id}.json",
    )


def in_scope_class_labels(train_records: Sequence[Sequence[str]]) -> list[str]:
    labels = sorted({record[1] for record in train_records})
    if len(labels) != B2_NUM_LABELS:
        raise ValueError(f"B2 requires {B2_NUM_LABELS} in-scope labels, found {len(labels)}")
    return labels


def verify_loaded_bert_config(config: Any) -> None:
    """Reject anything that is not the frozen BERT-base classification config."""
    model_type = getattr(config, "model_type", None)
    if model_type != "bert":
        raise ValueError(f"B2 requires a BERT encoder, found model_type={model_type!r}")
    if getattr(config, "is_decoder", False):
        raise ValueError("B2 requires a bidirectional encoder, found is_decoder=True")
    expected = {
        "hidden_size": B2_HIDDEN_SIZE,
        "num_hidden_layers": B2_NUM_HIDDEN_LAYERS,
        "num_attention_heads": B2_NUM_ATTENTION_HEADS,
        "vocab_size": B2_VOCAB_SIZE,
        "max_position_embeddings": B2_MAX_POSITION_EMBEDDINGS,
        "num_labels": B2_NUM_LABELS,
    }
    observed = {name: getattr(config, name, None) for name in expected}
    if observed != expected:
        raise ValueError(f"loaded BERT config does not match the B2 freeze: {observed}")


def verify_b2_probability_columns(
    probabilities: NDArray[np.float64], class_labels: Sequence[str]
) -> None:
    if probabilities.ndim != 2:
        raise ValueError("B2 probabilities must be a 2D matrix")
    if list(class_labels) != sorted(class_labels) or len(set(class_labels)) != len(class_labels):
        raise ValueError("B2 class_labels must be unique and in canonical sorted order")
    if probabilities.shape[1] != len(class_labels) or len(class_labels) != B2_NUM_LABELS:
        raise ValueError(
            "B2 probability columns must correspond exactly to the 150 canonical class labels"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("B2 probabilities contain non-finite values")


def configure_b2_determinism(seed: int) -> dict[str, object]:
    """Seed RNGs and request deterministic kernels; record what actually applied.

    PYTHONHASHSEED cannot be applied to an already-running interpreter. The
    startup value is recorded but is not treated as an in-process guarantee.
    """
    seed = require_allowed_b2_seed(seed)
    pythonhashseed_startup = os.environ.get("PYTHONHASHSEED")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    deterministic_algorithms = False
    deterministic_algorithms_error: str | None = None
    try:
        torch.use_deterministic_algorithms(True)
        deterministic_algorithms = True
    except (RuntimeError, TypeError) as error:
        deterministic_algorithms_error = str(error)
    return {
        "seed": seed,
        "cuda_available": cuda_available,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": deterministic_algorithms,
        "deterministic_algorithms_error": deterministic_algorithms_error,
        "pythonhashseed_startup": pythonhashseed_startup,
        "pythonhashseed_applied_in_process": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def resolve_b2_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_frozen_b2_model_and_tokenizer() -> tuple[Any, Any]:
    """Load only the pinned Hub revision. Never follow main."""
    config = AutoConfig.from_pretrained(
        B2_MODEL_ID,
        revision=B2_MODEL_REVISION,
        num_labels=B2_NUM_LABELS,
    )
    verify_loaded_bert_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        B2_TOKENIZER_ID,
        revision=B2_TOKENIZER_REVISION,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        B2_MODEL_ID,
        revision=B2_MODEL_REVISION,
        config=config,
    )
    verify_loaded_bert_config(model.config)
    return model, tokenizer


def _linear_warmup_decay_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 1.0 - progress)


def _encode_texts(tokenizer: Any, texts: Sequence[str]) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=B2_MAX_LENGTH,
        return_tensors="pt",
    )
    return {key: value for key, value in encoded.items()}


def _tensor_dataset(
    encodings: Mapping[str, torch.Tensor], labels: NDArray[np.int64]
) -> TensorDataset:
    tensors = [
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.as_tensor(labels, dtype=torch.long),
    ]
    if "token_type_ids" in encodings:
        tensors.insert(2, encodings["token_type_ids"])
    return TensorDataset(*tensors)


def _batch_to_inputs(
    batch: tuple[torch.Tensor, ...], device: torch.device, *, include_labels: bool
) -> dict[str, torch.Tensor]:
    if len(batch) == 4:
        input_ids, attention_mask, token_type_ids, labels = batch
        payload = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "token_type_ids": token_type_ids.to(device),
        }
    else:
        input_ids, attention_mask, labels = batch
        payload = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
    if include_labels:
        payload["labels"] = labels.to(device)
    return payload


def count_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def train_frozen_b2(
    *,
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    labels: NDArray[np.int64],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    """Train exactly three epochs and keep the final-epoch weights."""
    seed = require_allowed_b2_seed(seed)
    if len(texts) != labels.size:
        raise ValueError("B2 train texts and labels must align")
    model.to(device)
    model.train()
    encodings = _encode_texts(tokenizer, texts)
    dataset = _tensor_dataset(encodings, labels)
    steps_per_epoch = math.ceil(len(texts) / B2_BATCH_SIZE)
    total_steps = steps_per_epoch * B2_EPOCHS
    warmup_steps = int(total_steps * B2_WARMUP_RATIO)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=B2_LEARNING_RATE,
        betas=B2_ADAMW_BETAS,
        eps=B2_ADAMW_EPS,
        weight_decay=B2_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _linear_warmup_decay_lambda(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp) if use_cuda_amp else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(B2_EPOCHS):
            generator = torch.Generator()
            generator.manual_seed(seed + epoch)
            loader = DataLoader(
                dataset,
                batch_size=B2_BATCH_SIZE,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                generator=generator,
            )
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                inputs = _batch_to_inputs(batch, device, include_labels=True)
                if use_cuda_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = model(**inputs).loss
                    if scaler is None:
                        raise RuntimeError("CUDA AMP requested without a GradScaler")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), B2_MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model(**inputs).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), B2_MAX_GRAD_NORM)
                    optimizer.step()
                scheduler.step()
    except RuntimeError as error:
        message = str(error).lower()
        if "out of memory" in message:
            raise RuntimeError(
                "B2 froze batch_size=16 and max_length=128; CUDA OOM requires human review "
                "and must not be worked around by changing the frozen configuration"
            ) from error
        raise

    peak_gpu_memory_bytes: int | None = None
    if device.type == "cuda":
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    return {
        "epochs": B2_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "parameter_count": count_parameters(model),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "amp_grad_scaler": use_cuda_amp,
    }


def evaluate_frozen_b2(
    *,
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    labels: NDArray[np.int64],
    class_labels: Sequence[str],
    device: torch.device,
) -> tuple[NDArray[np.int64], NDArray[np.float64], BaselineMetrics]:
    if len(texts) != labels.size:
        raise ValueError("B2 evaluation texts and labels must align")
    model.to(device)
    model.eval()
    encodings = _encode_texts(tokenizer, texts)
    dataset = _tensor_dataset(encodings, labels)
    loader = DataLoader(
        dataset,
        batch_size=B2_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    logit_batches: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for batch in loader:
                inputs = _batch_to_inputs(batch, device, include_labels=False)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(**inputs).logits
                else:
                    logits = model(**inputs).logits
                logit_batches.append(logits.float().cpu())
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise RuntimeError(
                "B2 froze batch_size=16 and max_length=128; CUDA OOM requires human review "
                "and must not be worked around by changing the frozen configuration"
            ) from error
        raise
    logits = torch.cat(logit_batches, dim=0)
    if logits.shape != (labels.size, B2_NUM_LABELS):
        raise ValueError(
            "B2 probability columns do not correspond to canonical class IDs: "
            f"expected {(labels.size, B2_NUM_LABELS)}, found {tuple(logits.shape)}"
        )
    probabilities = torch.softmax(logits, dim=-1).numpy().astype(np.float64, copy=False)
    verify_b2_probability_columns(probabilities, class_labels)
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    metrics = compute_baseline_metrics(labels, predictions, probabilities)
    return predictions, probabilities, metrics


def aggregate_b2_metrics(per_seed: Mapping[int, BaselineMetrics]) -> dict[str, object]:
    """Mean and sample std over the predeclared seed order. Never pick a best seed."""
    if tuple(sorted(per_seed)) != tuple(sorted(B2_SEEDS)) or set(per_seed) != set(B2_SEEDS):
        raise ValueError(f"B2 aggregation requires exactly seeds {B2_SEEDS}")
    ordered = [per_seed[seed] for seed in B2_SEEDS]
    values = {
        "macro_f1": [metrics.macro_f1 for metrics in ordered],
        "accuracy": [metrics.accuracy for metrics in ordered],
        "nll": [metrics.nll for metrics in ordered],
        "multiclass_brier": [metrics.multiclass_brier for metrics in ordered],
    }
    summary: dict[str, object] = {
        "seeds": list(B2_SEEDS),
        "n_seeds": len(B2_SEEDS),
        "headline_rule": "mean_and_sample_std_over_predeclared_seeds",
    }
    for name, series in values.items():
        array = np.asarray(series, dtype=np.float64)
        summary[f"mean_{name}"] = float(array.mean())
        summary[f"std_{name}"] = float(array.std(ddof=1))
    return summary


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_b2_hardware_software() -> dict[str, object]:
    """Record available machine identity; never invent missing fields."""
    cpu: str | None = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip() or None
                    break
    except OSError:
        cpu = platform.processor() or None
    ram_gb: float | None = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    ram_gb = round(int(line.split()[1]) / 1024 / 1024, 3)
                    break
    except OSError:
        ram_gb = None
    gpu: str | None = "none"
    cuda_version: str | None = None
    gpu_driver: str | None = None
    gpu_total_memory_bytes: int | None = None
    gpu_index: int | None = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cuda_version = getattr(torch.version, "cuda", None)
        gpu_index = 0
        try:
            gpu_total_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        except (AssertionError, OSError, RuntimeError):
            gpu_total_memory_bytes = None
        try:
            with open("/proc/driver/nvidia/version", encoding="utf-8") as handle:
                gpu_driver = handle.readline().strip() or None
        except OSError:
            gpu_driver = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "gpu": gpu,
        "ram_gb": ram_gb,
        "cuda_version": cuda_version,
        "gpu_driver": gpu_driver,
        "gpu_index": gpu_index,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
        "numpy": _package_version("numpy"),
        "device": device,
        "dtype": dtype,
        "cuda_used": device == "cuda",
        "batch_size": B2_BATCH_SIZE,
        "max_length": B2_MAX_LENGTH,
        "model_id": B2_MODEL_ID,
        "model_revision": B2_MODEL_REVISION,
        "tokenizer_revision": B2_TOKENIZER_REVISION,
        "headline_hardware_qualification": "UNVERIFIED",
        "headline_hardware_assumption": (
            "RTX 4060-class 8 GB GPU; BERT-base; max_length 128; batch 16; fp16. "
            "Not an automated hardware-equivalence test."
        ),
    }
