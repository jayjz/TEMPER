"""Execute one frozen-config EXP-0001 baseline without fitting calibration.

The default target is the frozen validation partition.  Final-test evaluation is
an explicit invocation and this script has no tuning or selection behavior.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from temper.baselines import MajorityBaseline, TfidfLogisticRegressionBaseline
from temper.contracts import DatasetRef, ExperimentManifest, HardwareRef, ModelRef
from temper.datasets import (
    EXP0001_ARCHIVE_SHA256,
    EXP0001_CANONICAL_SHA256,
    FrozenSplits,
    validate_clinc150_payload,
    validate_exp0001_splits,
    verify_exp0001_archive_claim,
    verify_exp0001_canonical_dataset,
)
from temper.evaluation import compute_baseline_metrics, write_prediction_artifact
from temper.evidence import sha256_file


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hash_file(path: Path) -> str:
    return sha256_file(path)


def _verify_canonical_dataset(dataset_path: Path, canonical_sha256: str) -> str:
    return verify_exp0001_canonical_dataset(dataset_path, claimed_canonical=canonical_sha256)


def _records(
    records: list[list[str]], indices: tuple[int, ...], label_to_id: dict[str, int]
) -> tuple[list[str], np.ndarray]:
    selected = [records[index] for index in indices]
    return [record[0] for record in selected], np.array(
        [label_to_id[record[1]] for record in selected], dtype=np.int64
    )


def _validate_loaded_splits(splits: FrozenSplits, payload: dict[str, list[list[str]]]) -> None:
    """Reject loaded metadata that violates the frozen EXP-0001 seed-42 split contract."""
    validate_exp0001_splits(splits, payload)


def _verify_b1_probability_class_order(
    model: TfidfLogisticRegressionBaseline, n_classes: int
) -> None:
    expected = np.arange(n_classes, dtype=np.int64)
    observed = np.asarray(model.classifier.classes_, dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError(
            "B1 probability columns do not correspond to canonical class IDs: "
            f"expected {expected.tolist()}, observed {observed.tolist()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="validated data_full.json path")
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--canonical-sha256", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=("B0", "B1"), required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    arguments = parser.parse_args()

    verify_exp0001_archive_claim(arguments.archive_sha256)
    _verify_canonical_dataset(arguments.dataset, arguments.canonical_sha256)
    payload = validate_clinc150_payload(json.loads(arguments.dataset.read_text(encoding="utf-8")))
    splits = FrozenSplits.read(arguments.splits)
    _validate_loaded_splits(splits, payload)
    labels = sorted({record[1] for record in payload["train"]})
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_texts, train_labels = _records(payload["train"], splits.train_indices, label_to_id)
    evaluation_records = payload["val"] if arguments.partition == "validation" else payload["test"]
    evaluation_indices = (
        splits.validation_indices if arguments.partition == "validation" else splits.test_indices
    )
    evaluation_texts, evaluation_labels = _records(
        evaluation_records, evaluation_indices, label_to_id
    )

    started = time.perf_counter()
    if arguments.baseline == "B0":
        model = MajorityBaseline().fit(train_labels, n_classes=len(labels))
        preprocessing: dict[str, object] = {"text": "none"}
        hyperparameters: dict[str, object] = {
            "strategy": "train-majority",
            "n_classes": len(labels),
        }
        model_ref = ModelRef(name="majority-class", revision="B0")
    else:
        model = TfidfLogisticRegressionBaseline(random_state=splits.seed).fit(
            train_texts, train_labels
        )
        preprocessing = {"analyzer": "word", "lowercase": True, "ngram_range": [1, 1], "norm": "l2"}
        hyperparameters = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": splits.seed,
        }
        model_ref = ModelRef(name="tfidf-logistic-regression", revision="B1")
    fit_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    probabilities = model.predict_proba(evaluation_texts)
    if arguments.baseline == "B1":
        _verify_b1_probability_class_order(model, len(labels))
    inference_seconds = time.perf_counter() - inference_started
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    metrics = compute_baseline_metrics(evaluation_labels, predictions, probabilities)

    run_id = f"EXP-0001-{arguments.baseline}-{arguments.partition}-seed-{splits.seed}"
    artifact = arguments.output / "results" / f"{run_id}.npz"
    manifest_path = arguments.output / "manifests" / f"{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    write_prediction_artifact(
        artifact,
        labels=evaluation_labels,
        predictions=predictions,
        probabilities=probabilities,
        class_labels=np.asarray(labels),
        metrics=metrics,
    )
    manifest = ExperimentManifest(
        experiment_id="EXP-0001",
        run_id=run_id,
        phase="P1",
        status="RUNNING",
        git_commit=_git_commit(),
        dataset=DatasetRef(
            name="CLINC150",
            version="full",
            source="UCI 570 / clinc/oos-eval",
            archive_sha256=EXP0001_ARCHIVE_SHA256,
            canonical_sha256=EXP0001_CANONICAL_SHA256,
            label_provenance="published benchmark labels",
        ),
        model=model_ref,
        hardware=HardwareRef(platform=platform.platform()),
        split_definition=(
            "official train; official validation frozen into validation/calibration "
            f"seed={splits.seed}; official test untouched during fit"
        ),
        preprocessing=preprocessing,
        objective="multiclass classification",
        seed=splits.seed,
        hyperparameters=hyperparameters,
        software_environment={"python": sys.version.split()[0], "platform": platform.platform()},
        calibration_method=None,
        threshold_selection=None,
        metrics=("accuracy", "macro_f1", "nll", "multiclass_brier"),
        runtime={
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "n_examples": len(evaluation_texts),
        },
        artifact_paths=(artifact,),
        limitations=(
            "No calibration transform was fitted.",
            "CLINC150 benchmark labels are not production outcomes.",
        ),
        conclusion=None,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
