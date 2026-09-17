"""Fail-closed provenance checks for EXP-0001 B2 seed artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from temper.baselines.encoder import (
    B2_MODEL_ID,
    B2_MODEL_REVISION,
    B2_N_VALIDATION_EXAMPLES,
    B2_NUM_LABELS,
    B2_PARTITION,
    B2_SEEDS,
    B2_TOKENIZER_REVISION,
    aggregate_b2_metrics,
    b2_run_artifacts,
    frozen_b2_hyperparameters,
    verify_b2_probability_columns,
)
from temper.contracts import ExperimentManifest
from temper.datasets import EXP0001_ARCHIVE_SHA256, EXP0001_CANONICAL_SHA256
from temper.evaluation import BaselineMetrics, compute_baseline_metrics, read_prediction_artifact
from temper.evidence import verify_manifest_sidecar


def _require_equal(name: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ValueError(f"B2 {name} mismatch: expected {expected!r}, found {observed!r}")


def _require_integer_class_ids(name: str, values: NDArray[Any]) -> NDArray[np.int64]:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"B2 {name} must be integer class IDs")
    cast = values.astype(np.int64, copy=False)
    if (cast < 0).any() or (cast >= B2_NUM_LABELS).any():
        raise ValueError(f"B2 {name} must be class IDs in 0..{B2_NUM_LABELS - 1}")
    return cast


def verify_b2_prediction_arrays(
    labels: NDArray[Any],
    predictions: NDArray[Any],
    probabilities: NDArray[Any],
    class_labels: NDArray[Any],
) -> None:
    """Reject internally inconsistent B2 prediction artifacts."""
    if labels.shape != (B2_N_VALIDATION_EXAMPLES,):
        raise ValueError(
            f"B2 labels must have shape ({B2_N_VALIDATION_EXAMPLES},), found {labels.shape}"
        )
    if predictions.shape != (B2_N_VALIDATION_EXAMPLES,):
        raise ValueError(
            "B2 predictions must have shape "
            f"({B2_N_VALIDATION_EXAMPLES},), found {predictions.shape}"
        )
    expected_prob = (B2_N_VALIDATION_EXAMPLES, B2_NUM_LABELS)
    if probabilities.shape != expected_prob:
        raise ValueError(
            f"B2 probabilities must have shape {expected_prob}, found {probabilities.shape}"
        )
    if class_labels.shape != (B2_NUM_LABELS,):
        raise ValueError(
            f"B2 class_labels must have shape ({B2_NUM_LABELS},), found {class_labels.shape}"
        )
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise ValueError("B2 probabilities must be floating-point")
    if not np.isfinite(probabilities).all():
        raise ValueError("B2 probabilities contain non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("B2 probability rows must sum to 1")
    _require_integer_class_ids("labels", labels)
    predictions_i = _require_integer_class_ids("predictions", predictions)
    argmax = probabilities.argmax(axis=1).astype(np.int64, copy=False)
    if not np.array_equal(predictions_i, argmax):
        raise ValueError("B2 predictions must equal argmax(probabilities, axis=1)")
    class_label_list = [str(label) for label in class_labels.tolist()]
    verify_b2_probability_columns(probabilities.astype(np.float64, copy=False), class_label_list)


def _require_common_producer_commit(commits: list[object]) -> str:
    if any(not isinstance(commit, str) or not commit for commit in commits):
        raise ValueError("B2 aggregation requires a non-empty git_commit on every seed")
    unique = {str(commit) for commit in commits}
    if len(unique) != 1:
        raise ValueError(f"B2 aggregation requires one producer commit; found {sorted(unique)}")
    return str(commits[0])


def verify_b2_seed_bundle(output: Path, seed: int) -> dict[str, Any]:
    """Admit one seed only if its manifest, sidecar, and npz form a closed chain."""
    names = b2_run_artifacts(seed)
    artifact = output / "results" / names.result_name
    manifest_path = output / "manifests" / names.manifest_name
    if not artifact.exists():
        raise FileNotFoundError(f"missing B2 artifact for seed {seed}: {artifact}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing B2 manifest for seed {seed}: {manifest_path}")
    manifest = ExperimentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    artifact_sha256 = manifest.runtime.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or not artifact_sha256:
        raise ValueError(f"B2 manifest for seed {seed} is missing runtime.artifact_sha256")
    sidecar = verify_manifest_sidecar(
        manifest_path=manifest_path,
        artifact_path=artifact,
        expected_artifact_sha256=artifact_sha256,
    )
    _require_equal("experiment_id", manifest.experiment_id, "EXP-0001")
    _require_equal("run_id", manifest.run_id, names.run_id)
    _require_equal("seed", manifest.seed, seed)
    _require_equal("model.name", manifest.model.name, B2_MODEL_ID)
    _require_equal("model.revision", manifest.model.revision, B2_MODEL_REVISION)
    _require_equal(
        "model.tokenizer_revision",
        manifest.model.tokenizer_revision,
        B2_TOKENIZER_REVISION,
    )
    _require_equal(
        "dataset.canonical_sha256",
        manifest.dataset.canonical_sha256,
        EXP0001_CANONICAL_SHA256,
    )
    _require_equal(
        "dataset.archive_sha256",
        manifest.dataset.archive_sha256,
        EXP0001_ARCHIVE_SHA256,
    )
    _require_equal("calibration_method", manifest.calibration_method, None)
    _require_equal("threshold_selection", manifest.threshold_selection, None)
    _require_equal("hyperparameters", manifest.hyperparameters, frozen_b2_hyperparameters())
    _require_equal("partition", manifest.hyperparameters.get("partition"), B2_PARTITION)
    labels, predictions, probabilities, class_labels, stored_metrics = read_prediction_artifact(
        artifact
    )
    verify_b2_prediction_arrays(labels, predictions, probabilities, class_labels)
    recomputed = compute_baseline_metrics(labels, predictions, probabilities)
    if recomputed != stored_metrics:
        raise ValueError(
            f"B2 stored metrics for seed {seed} do not match recomputation from raw arrays"
        )
    return {
        "seed": seed,
        "run_id": names.run_id,
        "manifest_path": manifest_path,
        "artifact_path": artifact,
        "manifest_sha256": sidecar["manifest_sha256"],
        "artifact_sha256": sidecar["artifact_sha256"],
        "git_commit": manifest.git_commit,
        "metrics": recomputed,
    }


def load_verified_b2_metrics(output: Path) -> dict[int, BaselineMetrics]:
    """Load all and only the predeclared B2 seeds after provenance verification."""
    expected_names = {b2_run_artifacts(seed).result_name for seed in B2_SEEDS}
    results_dir = output / "results"
    extra_npz = sorted(
        path.name
        for path in results_dir.glob("EXP-0001-B2-*.npz")
        if path.name not in expected_names
    )
    if extra_npz:
        raise ValueError(f"unexpected extra B2 artifacts present: {extra_npz}")
    per_seed: dict[int, BaselineMetrics] = {}
    for seed in B2_SEEDS:
        metrics = verify_b2_seed_bundle(output, seed)["metrics"]
        if not isinstance(metrics, BaselineMetrics):
            raise TypeError(f"B2 seed {seed} did not yield BaselineMetrics")
        per_seed[seed] = metrics
    return per_seed


def verified_b2_aggregate(output: Path) -> dict[str, object]:
    bundles = [verify_b2_seed_bundle(output, seed) for seed in B2_SEEDS]
    extra_check = load_verified_b2_metrics(output)
    if set(extra_check) != set(B2_SEEDS):
        raise ValueError(f"B2 aggregation requires exactly seeds {B2_SEEDS}")
    producer_git_commit = _require_common_producer_commit(
        [bundle["git_commit"] for bundle in bundles]
    )
    summary = aggregate_b2_metrics({int(bundle["seed"]): bundle["metrics"] for bundle in bundles})
    return {
        "experiment_id": "EXP-0001",
        "baseline": "B2",
        "partition": B2_PARTITION,
        "headline_rule": summary["headline_rule"],
        "canonical_sha256": EXP0001_CANONICAL_SHA256,
        "archive_sha256": EXP0001_ARCHIVE_SHA256,
        "model_id": B2_MODEL_ID,
        "model_revision": B2_MODEL_REVISION,
        "tokenizer_revision": B2_TOKENIZER_REVISION,
        "producer_git_commit": producer_git_commit,
        "seeds": [
            {
                "seed": bundle["seed"],
                "run_id": bundle["run_id"],
                "artifact_sha256": bundle["artifact_sha256"],
                "manifest_sha256": bundle["manifest_sha256"],
                "git_commit": bundle["git_commit"],
            }
            for bundle in bundles
        ],
        "summary": summary,
    }
