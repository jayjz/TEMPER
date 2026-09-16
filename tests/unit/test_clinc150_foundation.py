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


def _normalized_payload_bytes(payload: dict[str, list[list[str]]]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)


_UCI_APPLEDOUBLE = (
    b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        \x00\x02\x00\x00"
    b"\x00\t\x00\x00\x00\x32\x00\x00\x00\x32\x00\x00\x00\x00\x00\x00\x00\x00"
)
_UCI_VALID_MEMBER = "clinc150_uci/data_full.json"
_UCI_APPLEDOUBLE_MEMBER = "__MACOSX/clinc150_uci/._data_full.json"


def test_acquisition_records_distinct_archive_and_canonical_hashes(tmp_path: Path) -> None:
    payload = _valid_payload()
    archive_path = tmp_path / "clinc150.zip"
    _write_zip(archive_path, [("data_full.json", json.dumps(payload, indent=2).encode("utf-8"))])

    dataset = acquire_clinc150(tmp_path)
    normalized_bytes = _normalized_payload_bytes(payload)

    assert dataset.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert dataset.canonical_sha256 == hashlib.sha256(dataset.path.read_bytes()).hexdigest()
    assert dataset.path.read_bytes() == normalized_bytes
    assert dataset.archive_sha256 != dataset.canonical_sha256


def test_single_valid_candidate_succeeds(tmp_path: Path) -> None:
    payload = _valid_payload()
    _write_zip(
        tmp_path / "clinc150.zip",
        [(_UCI_VALID_MEMBER, json.dumps(payload).encode("utf-8"))],
    )

    dataset = acquire_clinc150(tmp_path)

    assert dataset.path.read_bytes() == _normalized_payload_bytes(payload)
    assert dataset.canonical_sha256 == hashlib.sha256(dataset.path.read_bytes()).hexdigest()


def test_uci_archive_layout_is_handled_deterministically(tmp_path: Path) -> None:
    payload = _valid_payload()
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    archive_path = tmp_path / "clinc150.zip"
    _write_zip(
        archive_path,
        [
            (_UCI_APPLEDOUBLE_MEMBER, _UCI_APPLEDOUBLE),
            (_UCI_VALID_MEMBER, encoded),
            ("clinc150_uci/data_small.json", b'{"unrelated": true}'),
            ("__MACOSX/clinc150_uci/._data_small.json", _UCI_APPLEDOUBLE),
        ],
    )

    dataset = acquire_clinc150(tmp_path)
    normalized_bytes = _normalized_payload_bytes(payload)

    assert dataset.path.read_bytes() == normalized_bytes
    assert dataset.canonical_sha256 == hashlib.sha256(normalized_bytes).hexdigest()
    assert dataset.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert dataset.archive_sha256 != dataset.canonical_sha256


def test_duplicate_identical_valid_payloads_are_not_ambiguous(tmp_path: Path) -> None:
    payload = _valid_payload()
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    _write_zip(
        tmp_path / "clinc150.zip",
        [
            ("copy-a/data_full.json", encoded),
            ("copy-b/data_full.json", encoded),
        ],
    )

    dataset = acquire_clinc150(tmp_path)

    assert dataset.path.read_bytes() == _normalized_payload_bytes(payload)


def test_multiple_distinct_valid_payloads_fail_closed(tmp_path: Path) -> None:
    first = _valid_payload()
    second = _valid_payload()
    second["train"][0][0] = "altered-train-utterance"
    _write_zip(
        tmp_path / "clinc150.zip",
        [
            ("a/data_full.json", json.dumps(first).encode("utf-8")),
            ("b/data_full.json", json.dumps(second).encode("utf-8")),
        ],
    )

    with pytest.raises(ValueError, match=r"multiple distinct valid data_full\.json payloads"):
        acquire_clinc150(tmp_path)


def test_invalid_lookalike_data_full_json_files_are_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    destination = tmp_path / "valid-plus-lookalike"
    destination.mkdir()
    _write_zip(
        destination / "clinc150.zip",
        [
            (_UCI_VALID_MEMBER, json.dumps(payload).encode("utf-8")),
            (_UCI_APPLEDOUBLE_MEMBER, _UCI_APPLEDOUBLE),
            ("invalid/data_full.json", b'{"train": []}\n'),
        ],
    )

    dataset = acquire_clinc150(destination)
    assert dataset.path.read_bytes() == _normalized_payload_bytes(payload)

    lookalike_only = tmp_path / "lookalike-only"
    lookalike_only.mkdir()
    _write_zip(
        lookalike_only / "clinc150.zip",
        [
            (_UCI_APPLEDOUBLE_MEMBER, _UCI_APPLEDOUBLE),
            ("invalid/data_full.json", b'{"train": []}\n'),
        ],
    )
    with pytest.raises(ValueError, match=r"no schema-valid data_full\.json"):
        acquire_clinc150(lookalike_only)


def test_zip_member_order_does_not_change_canonical_content(tmp_path: Path) -> None:
    payload = _valid_payload()
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    lookalike = ("invalid/data_full.json", b'{"not": "clinc150"}')
    valid = (_UCI_VALID_MEMBER, encoded)
    appledouble = (_UCI_APPLEDOUBLE_MEMBER, _UCI_APPLEDOUBLE)

    first_dir = tmp_path / "order-a"
    second_dir = tmp_path / "order-b"
    first_dir.mkdir()
    second_dir.mkdir()
    _write_zip(first_dir / "clinc150.zip", [appledouble, lookalike, valid])
    _write_zip(second_dir / "clinc150.zip", [valid, lookalike, appledouble])

    first = acquire_clinc150(first_dir)
    second = acquire_clinc150(second_dir)
    expected = _normalized_payload_bytes(payload)

    assert first.path.read_bytes() == expected
    assert second.path.read_bytes() == expected
    assert first.canonical_sha256 == second.canonical_sha256 == hashlib.sha256(expected).hexdigest()
    assert first.archive_sha256 != second.archive_sha256
    assert first.archive_sha256 != first.canonical_sha256
    assert second.archive_sha256 != second.canonical_sha256


def test_malformed_zip_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "clinc150.zip").write_bytes(b"this is not a zip archive")

    with pytest.raises(ValueError, match="not a valid zip file"):
        acquire_clinc150(tmp_path)


def test_archive_without_data_full_json_fails_closed(tmp_path: Path) -> None:
    _write_zip(tmp_path / "clinc150.zip", [("readme.txt", b"no dataset here")])

    with pytest.raises(ValueError, match=r"must contain a data_full\.json member"):
        acquire_clinc150(tmp_path)


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
