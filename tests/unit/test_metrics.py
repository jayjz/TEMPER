import math

import numpy as np
import pytest

from temper.evaluation import (
    coverage,
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood,
    risk_coverage_curve,
    selective_risk,
)


def test_multiclass_brier_perfect_prediction_is_zero() -> None:
    probabilities = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1], dtype=np.int64)

    assert multiclass_brier_score(probabilities, labels) == pytest.approx(0.0)


def test_multiclass_brier_hand_calculated_example() -> None:
    probabilities = np.array([[0.75, 0.25]], dtype=np.float64)
    labels = np.array([0], dtype=np.int64)

    expected = (0.75 - 1.0) ** 2 + (0.25 - 0.0) ** 2

    assert multiclass_brier_score(probabilities, labels) == pytest.approx(expected)


def test_negative_log_likelihood_hand_calculated() -> None:
    probabilities = np.array([[0.8, 0.2]], dtype=np.float64)
    labels = np.array([0], dtype=np.int64)

    assert negative_log_likelihood(probabilities, labels) == pytest.approx(-math.log(0.8))


def test_coverage() -> None:
    confidences = np.array([0.95, 0.85, 0.60, 0.40], dtype=np.float64)

    assert coverage(confidences, 0.80) == pytest.approx(0.5)


def test_selective_risk() -> None:
    predictions = np.array([1, 0, 1, 0], dtype=np.int64)
    labels = np.array([1, 1, 1, 0], dtype=np.int64)
    confidences = np.array([0.95, 0.90, 0.60, 0.40], dtype=np.float64)

    assert selective_risk(
        predictions,
        labels,
        confidences,
        0.80,
    ) == pytest.approx(0.5)


def test_selective_risk_rejects_empty_acceptance_set() -> None:
    predictions = np.array([1], dtype=np.int64)
    labels = np.array([1], dtype=np.int64)
    confidences = np.array([0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="no samples accepted"):
        selective_risk(
            predictions,
            labels,
            confidences,
            0.9,
        )


def test_expected_calibration_error_hand_calculated() -> None:
    confidences = np.array([0.9, 0.8, 0.4, 0.3], dtype=np.float64)
    correctness = np.array([True, True, False, True], dtype=np.bool_)

    result = expected_calibration_error(
        confidences,
        correctness,
        n_bins=2,
    )

    expected = 0.5 * abs(0.5 - 0.35) + 0.5 * abs(1.0 - 0.85)

    assert result == pytest.approx(expected)


def test_risk_coverage_curve_hand_calculated() -> None:
    predictions = np.array([1, 0, 1, 0], dtype=np.int64)
    labels = np.array([1, 1, 1, 0], dtype=np.int64)
    confidences = np.array([0.95, 0.90, 0.60, 0.40], dtype=np.float64)

    coverage_values, risk_values = risk_coverage_curve(
        predictions,
        labels,
        confidences,
    )

    assert coverage_values.tolist() == pytest.approx([0.25, 0.50, 0.75, 1.0])

    assert risk_values.tolist() == pytest.approx([0.0, 0.5, 1.0 / 3.0, 0.25])
