import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from temper.baselines import MajorityBaseline, TfidfLogisticRegressionBaseline
from temper.datasets import (
    FrozenSplits,
    acquire_clinc150,
    freeze_clinc150_splits,
    validate_clinc150_payload,
)
from temper.evaluation import (
    compute_baseline_metrics,
    read_prediction_artifact,
    write_prediction_artifact,
)


def test_dataset_schema_rejects_unexpected_partition_structure() -> None:
    with pytest.raises(ValueError, match="partitions must be exactly"):
        validate_clinc150_payload({"train": []})


def _official_validation_labels() -> list[str]:
    return [f"intent-{label:03}" for label in range(150) for _ in range(20)]


def test_frozen_splits_are_stratified_reproducible_disjoint_and_preserve_test() -> None:
    labels = _official_validation_labels()
    first = freeze_clinc150_splits(validation_labels=labels, seed=7)
    second = freeze_clinc150_splits(validation_labels=labels, seed=7)
    alternate = freeze_clinc150_splits(validation_labels=labels, seed=8)

    assert first == second
    assert first != alternate
    assert Counter(labels[index] for index in first.validation_indices) == {
        label: 10 for label in set(labels)
    }
    assert Counter(labels[index] for index in first.calibration_indices) == {
        label: 10 for label in set(labels)
    }
    assert set(first.validation_indices).isdisjoint(first.calibration_indices)
    assert set(first.validation_indices) | set(first.calibration_indices) == set(range(3_000))
    assert len(first.validation_indices) == 1_500
    assert len(first.calibration_indices) == 1_500
    assert first.train_indices == tuple(range(15_000))
    assert first.test_indices == tuple(range(4_500))
    assert all(
        index < len(labels) for index in first.validation_indices + first.calibration_indices
    )


def test_split_serialization_round_trip_and_refuses_overwrite(tmp_path: Path) -> None:
    splits = freeze_clinc150_splits(validation_labels=_official_validation_labels())
    path = tmp_path / "splits.json"
    splits.write(path)

    assert FrozenSplits.read(path) == splits
    with pytest.raises(FileExistsError):
        splits.write(path)


def _valid_payload() -> dict[str, list[list[str]]]:
    labels = [f"intent-{index:03}" for index in range(150)]
    return {
        "train": [[f"train-{label}-{index}", label] for label in labels for index in range(100)],
        "val": [[f"val-{label}-{index}", label] for label in labels for index in range(20)],
        "test": [[f"test-{label}-{index}", label] for label in labels for index in range(30)],
        "oos_train": [[f"oos-train-{index}", "oos"] for index in range(100)],
        "oos_val": [[f"oos-val-{index}", "oos"] for index in range(100)],
        "oos_test": [[f"oos-test-{index}", "oos"] for index in range(1_000)],
    }


def test_acquisition_records_distinct_archive_and_canonical_hashes(tmp_path: Path) -> None:
    payload = _valid_payload()
    archive_path = tmp_path / "clinc150.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data_full.json", json.dumps(payload, indent=2))

    dataset = acquire_clinc150(tmp_path)
    normalized_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")

    assert dataset.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert dataset.canonical_sha256 == hashlib.sha256(dataset.path.read_bytes()).hexdigest()
    assert dataset.path.read_bytes() == normalized_bytes
    assert dataset.archive_sha256 != dataset.canonical_sha256


def test_majority_baseline_has_degenerate_normalized_probabilities() -> None:
    baseline = MajorityBaseline().fit(np.array([1, 1, 0], dtype=np.int64), n_classes=3)
    probabilities = baseline.predict_proba(["one", "two"])

    assert baseline.predict(["one"]).tolist() == [1]
    assert probabilities.tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_logistic_regression_probability_shape_and_normalization() -> None:
    model = TfidfLogisticRegressionBaseline(random_state=7).fit(
        ["pay bill", "bill payment", "weather now", "forecast weather"],
        np.array([0, 0, 1, 1], dtype=np.int64),
    )
    probabilities = model.predict_proba(["pay the bill", "weather forecast"])

    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_prediction_artifact_round_trip_and_metric_integration(tmp_path: Path) -> None:
    labels = np.array([0, 1], dtype=np.int64)
    predictions = np.array([0, 1], dtype=np.int64)
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    metrics = compute_baseline_metrics(labels, predictions, probabilities)
    path = tmp_path / "predictions.npz"
    write_prediction_artifact(
        path,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        class_labels=np.array(["intent-a", "intent-b"]),
        metrics=metrics,
    )

    restored = read_prediction_artifact(path)
    assert np.array_equal(restored[0], labels)
    assert np.array_equal(restored[1], predictions)
    assert np.array_equal(restored[2], probabilities)
    assert restored[3].tolist() == ["intent-a", "intent-b"]
    assert restored[4] == metrics
    assert metrics.accuracy == pytest.approx(1.0)
