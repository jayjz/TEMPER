"""Deterministic, serializable CLINC150 partition freezing."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrozenSplits:
    seed: int
    validation_fraction: float
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    calibration_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def write(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen split metadata: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> FrozenSplits:
        data = json.loads(path.read_text(encoding="utf-8"))
        for name in (
            "train_indices",
            "validation_indices",
            "calibration_indices",
            "test_indices",
        ):
            data[name] = tuple(data[name])
        return cls(**data)


def validate_exp0001_splits(
    splits: FrozenSplits, payload: dict[str, list[list[str]]]
) -> FrozenSplits:
    """Reject loaded metadata that violates the frozen EXP-0001 seed-42 split contract."""
    if splits.seed != 42:
        raise ValueError(f"EXP-0001 requires frozen split seed 42, found {splits.seed}")
    expected_sizes = {
        "train_indices": 15_000,
        "validation_indices": 1_500,
        "calibration_indices": 1_500,
        "test_indices": 4_500,
    }
    for name, expected_size in expected_sizes.items():
        indices = getattr(splits, name)
        if len(indices) != expected_size:
            raise ValueError(f"{name} has {len(indices)}, expected {expected_size}")

    def require_exact_indices(name: str, indices: tuple[int, ...], expected_size: int) -> None:
        expected = set(range(expected_size))
        if set(indices) != expected or len(set(indices)) != len(indices):
            raise ValueError(f"{name} must contain every official partition index exactly once")

    require_exact_indices("train_indices", splits.train_indices, len(payload["train"]))
    require_exact_indices("test_indices", splits.test_indices, len(payload["test"]))
    validation_space = set(range(len(payload["val"])))
    validation_indices = set(splits.validation_indices)
    calibration_indices = set(splits.calibration_indices)
    if not validation_indices <= validation_space or not calibration_indices <= validation_space:
        raise ValueError("validation and calibration indices must be official-validation indices")
    if validation_indices & calibration_indices:
        raise ValueError("validation and calibration indices must be disjoint")
    if validation_indices | calibration_indices != validation_space:
        raise ValueError(
            "validation and calibration indices must cover official validation exactly"
        )
    if len(validation_indices) != len(splits.validation_indices) or len(calibration_indices) != len(
        splits.calibration_indices
    ):
        raise ValueError("validation and calibration indices must be unique")

    validation_labels = [payload["val"][index][1] for index in splits.validation_indices]
    calibration_labels = [payload["val"][index][1] for index in splits.calibration_indices]
    expected_balance = {label: 10 for label in {record[1] for record in payload["val"]}}
    if Counter(validation_labels) != expected_balance:
        raise ValueError("validation indices must contain exactly 10 records from each class")
    if Counter(calibration_labels) != expected_balance:
        raise ValueError("calibration indices must contain exactly 10 records from each class")
    return splits


def freeze_clinc150_splits(
    *,
    validation_labels: Sequence[str],
    test_size: int = 4_500,
    seed: int = 42,
) -> FrozenSplits:
    """Freeze a 10/10-per-class split of official validation records only."""
    labels = tuple(validation_labels)
    label_to_indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)
    if len(label_to_indices) != 150 or any(
        len(indices) != 20 for indices in label_to_indices.values()
    ):
        raise ValueError("CLINC150 validation labels must contain 150 classes with 20 records each")

    rng = np.random.default_rng(seed)
    validation_indices: list[int] = []
    calibration_indices: list[int] = []
    for label in sorted(label_to_indices):
        permutation = rng.permutation(label_to_indices[label])
        calibration_indices.extend(int(index) for index in permutation[:10])
        validation_indices.extend(int(index) for index in permutation[10:])
    return FrozenSplits(
        seed=seed,
        validation_fraction=0.5,
        train_indices=tuple(range(15_000)),
        validation_indices=tuple(sorted(validation_indices)),
        calibration_indices=tuple(sorted(calibration_indices)),
        test_indices=tuple(range(test_size)),
    )
