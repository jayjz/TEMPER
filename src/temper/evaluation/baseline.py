"""Evaluation and artifact preservation for conventional probability classifiers."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score

from temper.evaluation.metrics import multiclass_brier_score, negative_log_likelihood
from temper.evidence import write_bytes_atomic


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    macro_f1: float
    nll: float
    multiclass_brier: float


def compute_baseline_metrics(
    labels: NDArray[np.int64], predictions: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> BaselineMetrics:
    return BaselineMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        nll=negative_log_likelihood(probabilities, labels),
        multiclass_brier=multiclass_brier_score(probabilities, labels),
    )


def write_prediction_artifact(
    path: Path,
    *,
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    class_labels: NDArray[np.str_],
    metrics: BaselineMetrics,
) -> None:
    """Atomically preserve raw evaluation outputs without overwriting prior runs."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prediction artifact: {path}")
    if labels.shape != predictions.shape or probabilities.shape[0] != labels.size:
        raise ValueError("artifact labels, predictions, and probabilities must align")
    if class_labels.ndim != 1 or class_labels.size != probabilities.shape[1]:
        raise ValueError("class_labels must provide one label for each probability column")
    if class_labels.size == 0 or len(set(class_labels.tolist())) != class_labels.size:
        raise ValueError("class_labels must be non-empty and unique")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        class_labels=class_labels,
        metrics=json.dumps(asdict(metrics)),
    )
    write_bytes_atomic(path, buffer.getvalue())


def read_prediction_artifact(
    path: Path,
) -> tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.str_], BaselineMetrics
]:
    with np.load(path, allow_pickle=False) as artifact:
        metrics = BaselineMetrics(**json.loads(str(artifact["metrics"])))
        return (
            artifact["labels"],
            artifact["predictions"],
            artifact["probabilities"],
            artifact["class_labels"],
            metrics,
        )
