from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExperimentStatus = Literal[
    "PLANNED",
    "RUNNING",
    "COMPLETE",
    "REPLICATED",
    "NEGATIVE",
    "INCONCLUSIVE",
    "INVALIDATED",
]


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    archive_sha256: str | None = None
    canonical_sha256: str | None = None
    label_provenance: str = Field(min_length=1)


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    revision: str | None = None
    tokenizer_revision: str | None = None


class HardwareRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: str = Field(min_length=1)
    cpu: str | None = None
    gpu: str | None = None
    ram_gb: float | None = Field(default=None, gt=0)


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(pattern=r"^EXP-\d{4}$")
    run_id: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    status: ExperimentStatus
    git_commit: str | None = None

    dataset: DatasetRef
    model: ModelRef
    hardware: HardwareRef

    split_definition: str = Field(min_length=1)
    preprocessing: dict[str, object]
    objective: str | None = None
    seed: int
    hyperparameters: dict[str, object]
    software_environment: dict[str, str]
    calibration_method: str | None = None
    threshold_selection: str | None = None
    metrics: tuple[str, ...]
    runtime: dict[str, object] = Field(default_factory=dict)
    artifact_paths: tuple[Path, ...] = ()
    limitations: tuple[str, ...] = ()
    conclusion: str | None = None
