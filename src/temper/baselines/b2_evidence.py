"""Fail-closed provenance checks for EXP-0001 B2 seed artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from temper.baselines.encoder import (
    B2_MODEL_ID,
    B2_MODEL_REVISION,
    B2_PARTITION,
    B2_SEEDS,
    B2_TOKENIZER_REVISION,
    aggregate_b2_metrics,
    b2_run_artifacts,
    frozen_b2_hyperparameters,
    verify_b2_probability_columns,
)
from temper.contracts import ExperimentManifest
from temper.datasets import EXP0001_CANONICAL_SHA256
from temper.evaluation import BaselineMetrics, compute_baseline_metrics, read_prediction_artifact
from temper.evidence import verify_manifest_sidecar


def _require_equal(name: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ValueError(f"B2 {name} mismatch: expected {expected!r}, found {observed!r}")


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
    _require_equal("calibration_method", manifest.calibration_method, None)
    _require_equal("threshold_selection", manifest.threshold_selection, None)
    _require_equal("hyperparameters", manifest.hyperparameters, frozen_b2_hyperparameters())
    _require_equal("partition", manifest.hyperparameters.get("partition"), B2_PARTITION)
    labels, predictions, probabilities, class_labels, stored_metrics = read_prediction_artifact(
        artifact
    )
    class_label_list = [str(label) for label in class_labels.tolist()]
    verify_b2_probability_columns(probabilities, class_label_list)
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
    summary = aggregate_b2_metrics({int(bundle["seed"]): bundle["metrics"] for bundle in bundles})
    return {
        "experiment_id": "EXP-0001",
        "baseline": "B2",
        "partition": B2_PARTITION,
        "headline_rule": summary["headline_rule"],
        "canonical_sha256": EXP0001_CANONICAL_SHA256,
        "model_id": B2_MODEL_ID,
        "model_revision": B2_MODEL_REVISION,
        "tokenizer_revision": B2_TOKENIZER_REVISION,
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
