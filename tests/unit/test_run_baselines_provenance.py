import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


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
