from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def multiclass_brier_score(
    probabilities: FloatArray,
    labels: IntArray,
) -> float:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")

    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must have matching rows")

    n_samples, n_classes = probabilities.shape

    if n_samples == 0:
        raise ValueError("at least one sample is required")

    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels contain an invalid class index")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must be between 0 and 1")

    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("probability rows must sum to 1")

    targets = np.zeros_like(probabilities)
    targets[np.arange(n_samples), labels] = 1.0

    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def negative_log_likelihood(
    probabilities: FloatArray,
    labels: IntArray,
    *,
    epsilon: float = 1e-12,
) -> float:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")

    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must have matching rows")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    selected = probabilities[np.arange(labels.shape[0]), labels]
    clipped = np.clip(selected, epsilon, 1.0)

    return float(-np.mean(np.log(clipped)))


def coverage(confidences: FloatArray, threshold: float) -> float:
    if confidences.ndim != 1:
        raise ValueError("confidences must be a 1D array")

    if confidences.size == 0:
        raise ValueError("at least one confidence value is required")

    return float(np.mean(confidences >= threshold))


def selective_risk(
    predictions: IntArray,
    labels: IntArray,
    confidences: FloatArray,
    threshold: float,
) -> float:
    if predictions.shape != labels.shape or predictions.shape != confidences.shape:
        raise ValueError("predictions, labels, and confidences must have matching shapes")

    accepted = confidences >= threshold

    if not np.any(accepted):
        raise ValueError("no samples accepted at this threshold")

    errors = predictions[accepted] != labels[accepted]

    return float(np.mean(errors))


def expected_calibration_error(
    confidences: FloatArray,
    correctness: NDArray[np.bool_],
    *,
    n_bins: int = 10,
) -> float:
    if confidences.ndim != 1:
        raise ValueError("confidences must be a 1D array")

    if correctness.ndim != 1:
        raise ValueError("correctness must be a 1D array")

    if confidences.shape != correctness.shape:
        raise ValueError("confidences and correctness must have matching shapes")

    if confidences.size == 0:
        raise ValueError("at least one sample is required")

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    if np.any(confidences < 0) or np.any(confidences > 1):
        raise ValueError("confidences must be between 0 and 1")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum(
        np.digitize(confidences, bin_edges[1:-1], right=True),
        n_bins - 1,
    )

    ece = 0.0

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index

        if not np.any(mask):
            continue

        bin_confidence = float(np.mean(confidences[mask]))
        bin_accuracy = float(np.mean(correctness[mask]))
        bin_weight = float(np.mean(mask))

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def risk_coverage_curve(
    predictions: IntArray,
    labels: IntArray,
    confidences: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if predictions.ndim != 1 or labels.ndim != 1 or confidences.ndim != 1:
        raise ValueError("predictions, labels, and confidences must be 1D arrays")

    if predictions.shape != labels.shape or predictions.shape != confidences.shape:
        raise ValueError("predictions, labels, and confidences must have matching shapes")

    if predictions.size == 0:
        raise ValueError("at least one sample is required")

    order = np.argsort(-confidences)
    sorted_predictions = predictions[order]
    sorted_labels = labels[order]

    errors = (sorted_predictions != sorted_labels).astype(np.float64)
    cumulative_errors = np.cumsum(errors)

    counts = np.arange(1, predictions.size + 1, dtype=np.float64)

    coverage_values = counts / predictions.size
    risk_values = cumulative_errors / counts

    return coverage_values, risk_values
