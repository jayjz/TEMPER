"""Execute one frozen EXP-0001 B2 seed on the validation partition only.

Calibration examples and the official test partition are not accepted.
Dataset identity is the frozen EXP-0001 canonical SHA-256, not a caller hash.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from temper.baselines.b2_evidence import verified_b2_aggregate
from temper.baselines.encoder import (
    B2_MAX_LENGTH,
    B2_MODEL_ID,
    B2_MODEL_REVISION,
    B2_PARTITION,
    B2_SEEDS,
    B2_TOKENIZER_REVISION,
    b2_run_artifacts,
    collect_b2_hardware_software,
    configure_b2_determinism,
    count_parameters,
    evaluate_frozen_b2,
    frozen_b2_hyperparameters,
    in_scope_class_labels,
    load_frozen_b2_model_and_tokenizer,
    require_allowed_b2_seed,
    require_b2_validation_partition,
    resolve_b2_device,
    train_frozen_b2,
)
from temper.contracts import DatasetRef, ExperimentManifest, HardwareRef, ModelRef
from temper.datasets import (
    EXP0001_ARCHIVE_SHA256,
    EXP0001_CANONICAL_SHA256,
    FrozenSplits,
    validate_clinc150_payload,
    validate_exp0001_splits,
    verify_exp0001_canonical_dataset,
)
from temper.evaluation import write_prediction_artifact
from temper.evidence import (
    require_clean_git,
    require_path_outside_repository,
    sha256_bytes,
    sha256_file,
    sidecar_digest_path,
    write_manifest_sidecar,
    write_text_atomic,
)


def _records(
    records: list[list[str]], indices: tuple[int, ...], label_to_id: dict[str, int]
) -> tuple[list[str], np.ndarray]:
    selected = [records[index] for index in indices]
    return [record[0] for record in selected], np.array(
        [label_to_id[record[1]] for record in selected], dtype=np.int64
    )


def _write_manifest(
    *,
    path: Path,
    run_id: str,
    seed: int,
    git_commit: str,
    hardware: HardwareRef,
    software_environment: dict[str, str],
    runtime: dict[str, object],
    artifact: Path,
    limitations: tuple[str, ...],
) -> str:
    sidecar_path = sidecar_digest_path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {path}")
    if sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest digest sidecar: {sidecar_path}")
    manifest = ExperimentManifest(
        experiment_id="EXP-0001",
        run_id=run_id,
        phase="P1",
        status="RUNNING",
        git_commit=git_commit,
        dataset=DatasetRef(
            name="CLINC150",
            version="full",
            source="UCI 570 / clinc/oos-eval",
            archive_sha256=EXP0001_ARCHIVE_SHA256,
            canonical_sha256=EXP0001_CANONICAL_SHA256,
            label_provenance="published benchmark labels",
        ),
        model=ModelRef(
            name=B2_MODEL_ID,
            revision=B2_MODEL_REVISION,
            tokenizer_revision=B2_TOKENIZER_REVISION,
        ),
        hardware=hardware,
        split_definition=(
            "official train; frozen seed-42 validation only; "
            "calibration unused; official test untouched"
        ),
        preprocessing={
            "tokenizer": B2_MODEL_ID,
            "tokenizer_revision": B2_TOKENIZER_REVISION,
            "truncation": True,
            "padding": "max_length",
            "max_length": B2_MAX_LENGTH,
        },
        objective="multiclass cross-entropy",
        seed=seed,
        hyperparameters=frozen_b2_hyperparameters(),
        software_environment=software_environment,
        calibration_method=None,
        threshold_selection=None,
        metrics=("accuracy", "macro_f1", "nll", "multiclass_brier"),
        runtime=runtime,
        artifact_paths=(artifact,),
        limitations=limitations,
        conclusion=None,
    )
    payload = manifest.model_dump_json(indent=2)
    manifest_sha256 = sha256_bytes(payload.encode("utf-8"))
    write_text_atomic(path, payload)
    write_manifest_sidecar(
        manifest_path=path,
        manifest_sha256=manifest_sha256,
        artifact_name=artifact.name,
        artifact_sha256=str(runtime["artifact_sha256"]),
    )
    return manifest_sha256


def _run_one_seed(arguments: argparse.Namespace) -> None:
    seed = require_allowed_b2_seed(arguments.seed)
    require_b2_validation_partition(B2_PARTITION)
    output = require_path_outside_repository(arguments.output)
    names = b2_run_artifacts(seed)
    artifact = output / "results" / names.result_name
    manifest_path = output / "manifests" / names.manifest_name
    sidecar_path = sidecar_digest_path(manifest_path)
    if artifact.exists():
        raise FileExistsError(f"refusing to overwrite prediction artifact: {artifact}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    if sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest digest sidecar: {sidecar_path}")

    git_commit = require_clean_git()
    observed_canonical = verify_exp0001_canonical_dataset(arguments.dataset)
    payload = validate_clinc150_payload(json.loads(arguments.dataset.read_text(encoding="utf-8")))
    splits = validate_exp0001_splits(FrozenSplits.read(arguments.splits), payload)
    class_labels = in_scope_class_labels(payload["train"])
    label_to_id = {label: index for index, label in enumerate(class_labels)}
    train_texts, train_labels = _records(payload["train"], splits.train_indices, label_to_id)
    validation_texts, validation_labels = _records(
        payload["val"], splits.validation_indices, label_to_id
    )

    snapshot = collect_b2_hardware_software()
    hardware = HardwareRef(
        platform=str(snapshot["platform"]),
        cpu=snapshot["cpu"] if isinstance(snapshot["cpu"], str) else None,
        gpu=snapshot["gpu"] if isinstance(snapshot["gpu"], str) else None,
        ram_gb=snapshot["ram_gb"] if isinstance(snapshot["ram_gb"], (int, float)) else None,
    )
    software_environment = {
        key: str(snapshot[key])
        for key in ("python", "torch", "transformers", "tokenizers", "numpy", "platform")
        if snapshot.get(key) is not None
    }
    determinism = configure_b2_determinism(seed)
    device = resolve_b2_device()
    dtype = "float16" if device.type == "cuda" else "float32"
    model, tokenizer = load_frozen_b2_model_and_tokenizer()
    determinism = configure_b2_determinism(seed)
    parameter_count = count_parameters(model)

    train_started = time.perf_counter()
    train_info = train_frozen_b2(
        model=model,
        tokenizer=tokenizer,
        texts=train_texts,
        labels=train_labels,
        seed=seed,
        device=device,
    )
    fit_seconds = time.perf_counter() - train_started
    inference_started = time.perf_counter()
    predictions, probabilities, metrics = evaluate_frozen_b2(
        model=model,
        tokenizer=tokenizer,
        texts=validation_texts,
        labels=validation_labels,
        class_labels=class_labels,
        device=device,
    )
    inference_seconds = time.perf_counter() - inference_started
    write_prediction_artifact(
        artifact,
        labels=validation_labels,
        predictions=predictions,
        probabilities=probabilities,
        class_labels=np.asarray(class_labels),
        metrics=metrics,
    )
    runtime: dict[str, object] = {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "n_examples": len(validation_texts),
        "throughput_examples_per_second": (
            len(validation_texts) / inference_seconds if inference_seconds else None
        ),
        "parameter_count": parameter_count,
        "device": str(device),
        "dtype": dtype,
        "cuda_used": device.type == "cuda",
        "gpu_index": snapshot.get("gpu_index"),
        "gpu_total_memory_bytes": snapshot.get("gpu_total_memory_bytes"),
        "headline_hardware_qualification": snapshot["headline_hardware_qualification"],
        "headline_hardware_assumption": snapshot["headline_hardware_assumption"],
        "batch_size": snapshot["batch_size"],
        "max_length": snapshot["max_length"],
        "cuda_version": snapshot["cuda_version"],
        "gpu_driver": snapshot["gpu_driver"],
        "mean_max_softmax": float(probabilities.max(axis=1).mean()),
        "artifact_sha256": sha256_file(artifact),
        "canonical_sha256_expected": EXP0001_CANONICAL_SHA256,
        "canonical_sha256_observed": observed_canonical,
        "canonical_sha256_verified": True,
        "archive_sha256_expected": EXP0001_ARCHIVE_SHA256,
        "archive_sha256_observed": None,
        "archive_sha256_source": "freeze_declared_not_rehashed",
        "git_dirty": False,
        **determinism,
        **train_info,
    }
    limitations = (
        "No calibration transform was fitted.",
        "CLINC150 benchmark labels are not production outcomes.",
        "Raw softmax probabilities are not calibrated.",
        "B2 does not use calibration or final-test examples.",
        "GPU runs are not claimed to be bit-identical.",
        "Headline hardware qualification is UNVERIFIED; CUDA is not RTX 4060-equivalent.",
        "Archive SHA-256 is freeze-declared and was not re-hashed from a zip this run.",
    )
    if device.type != "cuda":
        limitations = (
            *limitations,
            "CPU execution is not a headline GPU comparison under the B2 freeze.",
        )
    if determinism.get("deterministic_algorithms") is not True:
        limitations = (
            *limitations,
            "torch.use_deterministic_algorithms(True) did not fully apply; see runtime.",
        )
    manifest_sha256 = _write_manifest(
        path=manifest_path,
        run_id=names.run_id,
        seed=seed,
        git_commit=git_commit,
        hardware=hardware,
        software_environment=software_environment,
        runtime=runtime,
        artifact=artifact,
        limitations=limitations,
    )
    print(f"run_id={names.run_id}")
    print(f"artifact={artifact}")
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={manifest_sha256}")
    print(f"macro_f1={metrics.macro_f1}")
    print(f"accuracy={metrics.accuracy}")
    print(f"device={device}")
    print(f"headline_hardware_qualification={snapshot['headline_hardware_qualification']}")


def _aggregate(arguments: argparse.Namespace) -> None:
    output = require_path_outside_repository(arguments.output)
    summary = verified_b2_aggregate(output)
    summary_path = output / "manifests" / "EXP-0001-B2-validation-aggregate.json"
    sidecar_path = sidecar_digest_path(summary_path)
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate summary: {summary_path}")
    if sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate digest sidecar: {sidecar_path}")
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    write_text_atomic(summary_path, payload)
    write_manifest_sidecar(
        manifest_path=summary_path,
        manifest_sha256=sha256_bytes(payload.encode("utf-8")),
        artifact_name="EXP-0001-B2-validation-aggregate.json",
        artifact_sha256=sha256_bytes(payload.encode("utf-8")),
    )
    print(f"aggregate={summary_path}")
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--seed", type=int, choices=B2_SEEDS)
    arguments = parser.parse_args()
    if arguments.aggregate:
        _aggregate(arguments)
        return
    missing = [name for name in ("dataset", "splits", "seed") if getattr(arguments, name) is None]
    if missing:
        parser.error(f"the following arguments are required unless --aggregate: {missing}")
    _run_one_seed(arguments)


if __name__ == "__main__":
    main()
