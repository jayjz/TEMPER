import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from temper.datasets import FrozenSplits, freeze_clinc150_splits


@pytest.fixture
def run_baselines_module() -> ModuleType:
    path = Path("experiments/EXP-0001/run_baselines.py")
    specification = importlib.util.spec_from_file_location("run_baselines", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_hash_file_uses_exact_raw_dataset_bytes(
    run_baselines_module: ModuleType, tmp_path: Path
) -> None:
    compact_path = tmp_path / "compact.json"
    formatted_path = tmp_path / "formatted.json"
    compact_bytes = b'{"partition":["text","label"]}'
    formatted_bytes = b'{\n  "partition": ["text", "label"]\n}\n'
    compact_path.write_bytes(compact_bytes)
    formatted_path.write_bytes(formatted_bytes)

    assert (
        run_baselines_module._hash_file(compact_path) == hashlib.sha256(compact_bytes).hexdigest()
    )
    assert (
        run_baselines_module._hash_file(formatted_path)
        == hashlib.sha256(formatted_bytes).hexdigest()
    )
    assert run_baselines_module._hash_file(compact_path) != run_baselines_module._hash_file(
        formatted_path
    )


def test_matching_canonical_hash_continues_past_verification(
    monkeypatch: pytest.MonkeyPatch, run_baselines_module: ModuleType, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "data_full.json"
    dataset_path.write_bytes(b"{}")
    expected_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    def stop_after_verification(payload: object) -> dict[str, list[list[str]]]:
        assert payload == {}
        raise RuntimeError("verification passed")

    monkeypatch.setattr(run_baselines_module, "validate_clinc150_payload", stop_after_verification)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baselines.py",
            "--dataset",
            str(dataset_path),
            "--archive-sha256",
            "archive-provenance-only",
            "--canonical-sha256",
            expected_sha256,
            "--splits",
            str(tmp_path / "splits.json"),
            "--output",
            str(tmp_path / "output"),
            "--baseline",
            "B0",
        ],
    )

    with pytest.raises(RuntimeError, match="verification passed"):
        run_baselines_module.main()


def test_mismatched_canonical_hash_rejects_before_execution_or_artifacts(
    monkeypatch: pytest.MonkeyPatch, run_baselines_module: ModuleType, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "data_full.json"
    dataset_path.write_bytes(b"{}")
    output_path = tmp_path / "output"

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("execution advanced past dataset hash verification")

    monkeypatch.setattr(run_baselines_module, "validate_clinc150_payload", fail_if_called)
    monkeypatch.setattr(run_baselines_module, "MajorityBaseline", fail_if_called)
    monkeypatch.setattr(run_baselines_module, "TfidfLogisticRegressionBaseline", fail_if_called)
    monkeypatch.setattr(run_baselines_module, "compute_baseline_metrics", fail_if_called)
    monkeypatch.setattr(run_baselines_module, "write_prediction_artifact", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baselines.py",
            "--dataset",
            str(dataset_path),
            "--archive-sha256",
            "archive-provenance-only",
            "--canonical-sha256",
            "0" * 64,
            "--splits",
            str(tmp_path / "splits.json"),
            "--output",
            str(output_path),
            "--baseline",
            "B0",
        ],
    )

    observed_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    with pytest.raises(
        ValueError,
        match=f"expected {'0' * 64}, observed {observed_sha256}",
    ):
        run_baselines_module.main()

    assert not output_path.exists()


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


def _valid_splits() -> FrozenSplits:
    return freeze_clinc150_splits(
        validation_labels=[record[1] for record in _valid_payload()["val"]], seed=42
    )


def test_valid_seed_42_frozen_splits_pass_integrity_check(
    run_baselines_module: ModuleType,
) -> None:
    run_baselines_module._validate_loaded_splits(_valid_splits(), _valid_payload())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda splits: FrozenSplits(
                seed=splits.seed,
                validation_fraction=splits.validation_fraction,
                train_indices=splits.train_indices[:-1],
                validation_indices=splits.validation_indices,
                calibration_indices=splits.calibration_indices,
                test_indices=splits.test_indices,
            ),
            "train_indices has 14999, expected 15000",
        ),
        (
            lambda splits: FrozenSplits(
                seed=splits.seed,
                validation_fraction=splits.validation_fraction,
                train_indices=splits.train_indices,
                validation_indices=splits.validation_indices,
                calibration_indices=(splits.validation_indices[0], *splits.calibration_indices[1:]),
                test_indices=splits.test_indices,
            ),
            "validation and calibration indices must be disjoint",
        ),
        (
            lambda splits: FrozenSplits(
                seed=splits.seed,
                validation_fraction=splits.validation_fraction,
                train_indices=splits.train_indices,
                validation_indices=(
                    *splits.validation_indices[:-1],
                    splits.validation_indices[0],
                ),
                calibration_indices=splits.calibration_indices,
                test_indices=splits.test_indices,
            ),
            "validation and calibration indices must cover official validation exactly",
        ),
        (
            lambda splits: FrozenSplits(
                seed=splits.seed,
                validation_fraction=splits.validation_fraction,
                train_indices=splits.train_indices,
                validation_indices=(
                    *splits.validation_indices[1:],
                    splits.calibration_indices[100],
                ),
                calibration_indices=(
                    *splits.calibration_indices[:100],
                    splits.validation_indices[0],
                    *splits.calibration_indices[101:],
                ),
                test_indices=splits.test_indices,
            ),
            "validation indices must contain exactly 10 records from each class",
        ),
        (
            lambda splits: FrozenSplits(
                seed=7,
                validation_fraction=splits.validation_fraction,
                train_indices=splits.train_indices,
                validation_indices=splits.validation_indices,
                calibration_indices=splits.calibration_indices,
                test_indices=splits.test_indices,
            ),
            "EXP-0001 requires frozen split seed 42, found 7",
        ),
    ],
)
def test_invalid_loaded_splits_are_rejected(
    run_baselines_module: ModuleType,
    mutate: Callable[[FrozenSplits], FrozenSplits],
    message: str,
) -> None:
    splits = _valid_splits()
    invalid_splits = mutate(splits)
    with pytest.raises(ValueError, match=message):
        run_baselines_module._validate_loaded_splits(invalid_splits, _valid_payload())


def test_b1_rejects_noncanonical_probability_class_order(
    run_baselines_module: ModuleType,
) -> None:
    model = SimpleNamespace(classifier=SimpleNamespace(classes_=np.array([1, 0])))
    with pytest.raises(ValueError, match="do not correspond to canonical class IDs"):
        run_baselines_module._verify_b1_probability_class_order(model, n_classes=2)


def test_malformed_split_rejects_before_baseline_fitting(
    monkeypatch: pytest.MonkeyPatch, run_baselines_module: ModuleType, tmp_path: Path
) -> None:
    payload = _valid_payload()
    dataset_path = tmp_path / "data_full.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    splits = _valid_splits()
    malformed = FrozenSplits(
        seed=splits.seed,
        validation_fraction=splits.validation_fraction,
        train_indices=splits.train_indices[:-1],
        validation_indices=splits.validation_indices,
        calibration_indices=splits.calibration_indices,
        test_indices=splits.test_indices,
    )
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(malformed.__dict__), encoding="utf-8")

    def fail_if_fit_attempted(*args: object, **kwargs: object) -> None:
        raise AssertionError("baseline fitting must not begin for malformed splits")

    monkeypatch.setattr(run_baselines_module, "MajorityBaseline", fail_if_fit_attempted)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baselines.py",
            "--dataset",
            str(dataset_path),
            "--archive-sha256",
            "archive-provenance-only",
            "--canonical-sha256",
            hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "--splits",
            str(split_path),
            "--output",
            str(tmp_path / "output"),
            "--baseline",
            "B0",
        ],
    )

    with pytest.raises(ValueError, match="train_indices has 14999, expected 15000"):
        run_baselines_module.main()
