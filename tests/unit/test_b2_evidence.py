from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from temper.baselines.b2_evidence import verified_b2_aggregate, verify_b2_seed_bundle
from temper.baselines.encoder import (
    B2_MODEL_ID,
    B2_MODEL_REVISION,
    B2_SEEDS,
    B2_TOKENIZER_REVISION,
    b2_run_artifacts,
    frozen_b2_hyperparameters,
    require_b2_validation_partition,
)
from temper.contracts import DatasetRef, ExperimentManifest, HardwareRef, ModelRef
from temper.datasets import EXP0001_ARCHIVE_SHA256, EXP0001_CANONICAL_SHA256
from temper.evaluation import compute_baseline_metrics, write_prediction_artifact
from temper.evidence import (
    require_clean_git,
    sha256_bytes,
    sha256_file,
    sidecar_digest_path,
    verify_manifest_sidecar,
    write_manifest_sidecar,
    write_text_atomic,
)


@pytest.fixture
def run_b2_module() -> ModuleType:
    path = Path("experiments/EXP-0001/run_b2.py")
    specification = importlib.util.spec_from_file_location("run_b2_evidence", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _class_labels() -> list[str]:
    return [f"intent-{index:03}" for index in range(150)]


def _toy_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    probabilities = np.zeros((4, 150), dtype=np.float64)
    for index, label in enumerate(labels):
        probabilities[index, label] = 1.0
    predictions = labels.copy()
    class_labels = np.array(_class_labels())
    return labels, predictions, probabilities, class_labels


def _write_seed_bundle(
    output: Path,
    seed: int,
    *,
    model_revision: str = B2_MODEL_REVISION,
    canonical_sha256: str = EXP0001_CANONICAL_SHA256,
    manifest_seed: int | None = None,
    write_sidecar: bool = True,
    write_manifest: bool = True,
    write_artifact: bool = True,
) -> None:
    names = b2_run_artifacts(seed)
    labels, predictions, probabilities, class_labels = _toy_arrays()
    metrics = compute_baseline_metrics(labels, predictions, probabilities)
    artifact = output / "results" / names.result_name
    manifest_path = output / "manifests" / names.manifest_name
    if write_artifact:
        write_prediction_artifact(
            artifact,
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
            class_labels=class_labels,
            metrics=metrics,
        )
    if not write_manifest:
        return
    recorded_seed = seed if manifest_seed is None else manifest_seed
    manifest = ExperimentManifest(
        experiment_id="EXP-0001",
        run_id=names.run_id,
        phase="P1",
        status="RUNNING",
        git_commit="deadbeef",
        dataset=DatasetRef(
            name="CLINC150",
            version="full",
            source="UCI 570 / clinc/oos-eval",
            archive_sha256=EXP0001_ARCHIVE_SHA256,
            canonical_sha256=canonical_sha256,
            label_provenance="published benchmark labels",
        ),
        model=ModelRef(
            name=B2_MODEL_ID,
            revision=model_revision,
            tokenizer_revision=B2_TOKENIZER_REVISION,
        ),
        hardware=HardwareRef(platform="test"),
        split_definition="official train; frozen seed-42 validation only",
        preprocessing={"max_length": 128},
        objective="multiclass cross-entropy",
        seed=recorded_seed,
        hyperparameters=frozen_b2_hyperparameters(),
        software_environment={"python": "3.12"},
        calibration_method=None,
        threshold_selection=None,
        metrics=("accuracy", "macro_f1", "nll", "multiclass_brier"),
        runtime={"artifact_sha256": sha256_file(artifact) if artifact.exists() else "missing"},
        artifact_paths=(artifact,),
        limitations=("test fixture",),
        conclusion=None,
    )
    payload = manifest.model_dump_json(indent=2)
    write_text_atomic(manifest_path, payload)
    if write_sidecar and artifact.exists():
        write_manifest_sidecar(
            manifest_path=manifest_path,
            manifest_sha256=sha256_bytes(payload.encode("utf-8")),
            artifact_name=artifact.name,
            artifact_sha256=sha256_file(artifact),
        )


def _write_all_seeds(output: Path) -> None:
    for seed in B2_SEEDS:
        _write_seed_bundle(output, seed)


def _reject_before_load(
    run_b2_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_bytes: bytes,
) -> str:
    dataset_path = tmp_path / "data_full.json"
    dataset_path.write_bytes(dataset_bytes)
    loaded = {"called": False}

    def fail_if_loaded() -> tuple[object, object]:
        loaded["called"] = True
        raise AssertionError("model loaded before dataset identity failed")

    monkeypatch.setattr(run_b2_module, "require_clean_git", lambda: "abc")
    monkeypatch.setattr(run_b2_module, "load_frozen_b2_model_and_tokenizer", fail_if_loaded)
    arguments = argparse.Namespace(
        dataset=dataset_path,
        splits=tmp_path / "splits.json",
        output=tmp_path,
        seed=13,
    )
    with pytest.raises(ValueError, match="frozen EXP-0001 canonical identity"):
        run_b2_module._run_one_seed(arguments)
    assert loaded["called"] is False
    return hashlib.sha256(dataset_bytes).hexdigest()


def test_modified_dataset_with_matching_caller_hash_is_rejected_by_b2(
    run_b2_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = _reject_before_load(run_b2_module, monkeypatch, tmp_path, b'{"tampered": true}')
    assert observed != EXP0001_CANONICAL_SHA256


def test_wrong_frozen_canonical_hash_rejects_before_model_loading(
    run_b2_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reject_before_load(run_b2_module, monkeypatch, tmp_path, b"{}")


def test_mismatched_manifest_seed_is_rejected(tmp_path: Path) -> None:
    _write_seed_bundle(tmp_path, 13, manifest_seed=21)
    with pytest.raises(ValueError, match="B2 seed mismatch"):
        verify_b2_seed_bundle(tmp_path, 13)


def test_wrong_model_revision_is_rejected_during_aggregation(tmp_path: Path) -> None:
    _write_all_seeds(tmp_path)
    sidecar_digest_path(tmp_path / "manifests" / b2_run_artifacts(21).manifest_name).unlink()
    (tmp_path / "manifests" / b2_run_artifacts(21).manifest_name).unlink()
    (tmp_path / "results" / b2_run_artifacts(21).result_name).unlink()
    _write_seed_bundle(tmp_path, 21, model_revision="not-the-frozen-revision")
    with pytest.raises(ValueError, match=r"model\.revision"):
        verified_b2_aggregate(tmp_path)


def test_wrong_dataset_hash_in_manifest_is_rejected(tmp_path: Path) -> None:
    _write_all_seeds(tmp_path)
    sidecar_digest_path(tmp_path / "manifests" / b2_run_artifacts(13).manifest_name).unlink()
    (tmp_path / "manifests" / b2_run_artifacts(13).manifest_name).unlink()
    (tmp_path / "results" / b2_run_artifacts(13).result_name).unlink()
    _write_seed_bundle(tmp_path, 13, canonical_sha256="0" * 64)
    with pytest.raises(ValueError, match=r"dataset\.canonical_sha256"):
        verified_b2_aggregate(tmp_path)


def test_tampered_npz_after_manifest_generation_is_rejected(tmp_path: Path) -> None:
    _write_all_seeds(tmp_path)
    artifact = tmp_path / "results" / b2_run_artifacts(13).result_name
    artifact.write_bytes(artifact.read_bytes() + b"\x00tamper")
    with pytest.raises(ValueError, match="prediction artifact SHA-256"):
        verified_b2_aggregate(tmp_path)


def test_renamed_seed_artifact_cannot_masquerade(tmp_path: Path) -> None:
    _write_seed_bundle(tmp_path, 13)
    source = b2_run_artifacts(13)
    target = b2_run_artifacts(21)
    (tmp_path / "results" / target.result_name).write_bytes(
        (tmp_path / "results" / source.result_name).read_bytes()
    )
    (tmp_path / "manifests" / target.manifest_name).write_text(
        (tmp_path / "manifests" / source.manifest_name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    sidecar_digest_path(tmp_path / "manifests" / target.manifest_name).write_text(
        sidecar_digest_path(tmp_path / "manifests" / source.manifest_name).read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"digest sidecar|run_id|seed mismatch"):
        verify_b2_seed_bundle(tmp_path, 21)


def test_missing_manifest_or_missing_seed_fails_aggregation(tmp_path: Path) -> None:
    _write_seed_bundle(tmp_path, 13)
    _write_seed_bundle(tmp_path, 21)
    with pytest.raises(FileNotFoundError, match="missing B2 artifact for seed 37"):
        verified_b2_aggregate(tmp_path)
    _write_seed_bundle(tmp_path, 37, write_manifest=False)
    with pytest.raises(FileNotFoundError, match="missing B2 manifest for seed 37"):
        verified_b2_aggregate(tmp_path)


def test_aggregate_still_requires_exactly_the_predeclared_seeds(tmp_path: Path) -> None:
    _write_all_seeds(tmp_path)
    extra = tmp_path / "results" / "EXP-0001-B2-validation-seed-99.npz"
    extra.write_bytes((tmp_path / "results" / b2_run_artifacts(13).result_name).read_bytes())
    with pytest.raises(ValueError, match="unexpected extra B2 artifacts"):
        verified_b2_aggregate(tmp_path)


def test_overwrite_refusal_covers_manifest_sidecar_and_artifact(tmp_path: Path) -> None:
    _write_seed_bundle(tmp_path, 13)
    names = b2_run_artifacts(13)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_prediction_artifact(
            tmp_path / "results" / names.result_name,
            labels=np.array([0], dtype=np.int64),
            predictions=np.array([0], dtype=np.int64),
            probabilities=np.array([[1.0]], dtype=np.float64),
            class_labels=np.array(["a"]),
            metrics=compute_baseline_metrics(
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.array([[1.0]], dtype=np.float64),
            ),
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_text_atomic(tmp_path / "manifests" / names.manifest_name, "{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_manifest_sidecar(
            manifest_path=tmp_path / "manifests" / names.manifest_name,
            manifest_sha256="abc",
            artifact_name=names.result_name,
            artifact_sha256="def",
        )


def test_manifest_digest_sidecar_verifies_and_detects_tamper(tmp_path: Path) -> None:
    _write_seed_bundle(tmp_path, 13)
    names = b2_run_artifacts(13)
    manifest_path = tmp_path / "manifests" / names.manifest_name
    artifact = tmp_path / "results" / names.result_name
    sidecar = verify_manifest_sidecar(
        manifest_path=manifest_path,
        artifact_path=artifact,
        expected_artifact_sha256=sha256_file(artifact),
    )
    assert sidecar["manifest_sha256"] == sha256_file(manifest_path)
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256 does not match digest sidecar"):
        verify_manifest_sidecar(
            manifest_path=manifest_path,
            artifact_path=artifact,
            expected_artifact_sha256=sha256_file(artifact),
        )


def test_calibration_and_test_boundaries_remain_unchanged() -> None:
    source = Path("experiments/EXP-0001/run_b2.py").read_text(encoding="utf-8")
    assert "--partition" not in source
    assert 'payload["test"]' not in source
    assert "calibration_indices" not in source
    assert 'payload["val"]' in source
    with pytest.raises(ValueError, match="frozen validation partition"):
        require_b2_validation_partition("calibration")
    with pytest.raises(ValueError, match="frozen validation partition"):
        require_b2_validation_partition("test")


def test_verified_aggregate_does_not_select_best_seed(tmp_path: Path) -> None:
    _write_all_seeds(tmp_path)
    payload = verified_b2_aggregate(tmp_path)
    assert payload["summary"]["seeds"] == [13, 21, 37]
    assert "best_seed" not in payload
    assert "best_seed" not in payload["summary"]
    assert payload["canonical_sha256"] == EXP0001_CANONICAL_SHA256
    assert payload["model_revision"] == B2_MODEL_REVISION


def test_require_clean_git_fails_closed_when_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "temper.evidence.integrity.inspect_git_provenance",
        lambda: {"git_commit": "abc", "git_dirty": True, "git_status_porcelain": " M file"},
    )
    with pytest.raises(RuntimeError, match="dirty"):
        require_clean_git()
