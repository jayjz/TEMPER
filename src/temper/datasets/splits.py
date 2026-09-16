"""Deterministic, serializable CLINC150 partition freezing."""

from __future__ import annotations

import json
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
