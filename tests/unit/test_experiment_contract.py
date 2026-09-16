from pathlib import Path

import pytest
from pydantic import ValidationError

from temper.contracts import (
    DatasetRef,
    ExperimentManifest,
    HardwareRef,
    ModelRef,
)


def valid_manifest() -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id="EXP-0001",
        run_id="EXP-0001-B1-validation-seed-42",
        phase="P1",
        status="PLANNED",
        dataset=DatasetRef(
            name="CLINC150",
            version="full",
            source="UCI 570 / clinc/oos-eval",
            archive_sha256=None,
            canonical_sha256=None,
            label_provenance="published benchmark labels",
        ),
        model=ModelRef(name="tfidf-logistic-regression"),
        hardware=HardwareRef(
            platform="Windows",
            gpu="RTX 4060",
            ram_gb=32,
        ),
        split_definition=(
            "official train; validation deterministically split into "
            "validation/calibration; official test frozen"
        ),
        preprocessing={"text": "TF-IDF"},
        objective="multiclass classification",
        seed=42,
        hyperparameters={"random_state": 42},
        software_environment={"python": "3.12"},
        calibration_method=None,
        threshold_selection=None,
        metrics=("macro_f1", "accuracy", "brier_score", "nll"),
        runtime={"fit_seconds": 1.0},
        artifact_paths=(Path("experiments/EXP-0001"),),
        limitations=("CLINC150 is crowdsourced rather than production traffic.",),
    )


def test_valid_experiment_manifest() -> None:
    manifest = valid_manifest()

    assert manifest.experiment_id == "EXP-0001"
    assert manifest.dataset.name == "CLINC150"
    assert manifest.preprocessing["text"] == "TF-IDF"


def test_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetRef(
            name="example",
            version="1",
            source="public",
            label_provenance="benchmark",
            unexpected="nope",
        )


def test_manifest_rejects_bad_experiment_id() -> None:
    data = valid_manifest().model_dump()
    data["experiment_id"] = "experiment-1"

    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(data)


def test_manifest_is_immutable() -> None:
    manifest = valid_manifest()

    with pytest.raises(ValidationError):
        manifest.experiment_id = "EXP-9999"
