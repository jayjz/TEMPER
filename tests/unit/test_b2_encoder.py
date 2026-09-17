from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from temper.baselines.encoder import (
    B2_BATCH_SIZE,
    B2_EPOCHS,
    B2_MAX_LENGTH,
    B2_MODEL_ID,
    B2_MODEL_REVISION,
    B2_SEEDS,
    B2_TOKENIZER_REVISION,
    aggregate_b2_metrics,
    b2_run_artifacts,
    collect_b2_hardware_software,
    frozen_b2_hyperparameters,
    in_scope_class_labels,
    load_frozen_b2_model_and_tokenizer,
    require_allowed_b2_seed,
    require_b2_validation_partition,
    train_frozen_b2,
    verify_b2_probability_columns,
    verify_loaded_bert_config,
)
from temper.datasets import freeze_clinc150_splits, validate_exp0001_splits
from temper.evaluation import BaselineMetrics, write_prediction_artifact


@pytest.fixture
def run_b2_module() -> ModuleType:
    path = Path("experiments/EXP-0001/run_b2.py")
    specification = importlib.util.spec_from_file_location("run_b2", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _bert_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_type": "bert",
        "is_decoder": False,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "vocab_size": 30522,
        "max_position_embeddings": 512,
        "num_labels": 150,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _class_labels() -> list[str]:
    return [f"intent-{index:03}" for index in range(150)]


def test_frozen_config_serialization_contains_pinned_revisions_and_seeds() -> None:
    config = frozen_b2_hyperparameters()
    assert config["model_id"] == B2_MODEL_ID
    assert config["model_revision"] == B2_MODEL_REVISION
    assert B2_MODEL_REVISION == "86b5e0934494bd15c9632b12f734a8a67f723594"
    assert config["tokenizer_revision"] == B2_TOKENIZER_REVISION
    assert config["max_length"] == B2_MAX_LENGTH == 128
    assert config["batch_size"] == B2_BATCH_SIZE == 16
    assert config["epochs"] == B2_EPOCHS == 3
    assert config["gradient_accumulation"] is None
    assert config["early_stopping"] is None
    assert config["checkpoint_rule"] == "final-epoch-only"
    assert config["allowed_seeds"] == [13, 21, 37]
    assert config["partition"] == "validation"
    json.dumps(config)


def test_only_predeclared_b2_seeds_are_accepted() -> None:
    for seed in B2_SEEDS:
        assert require_allowed_b2_seed(seed) == seed
    with pytest.raises(ValueError, match=r"B2 seed must be one of"):
        require_allowed_b2_seed(42)
    with pytest.raises(ValueError, match=r"B2 seed must be one of"):
        require_allowed_b2_seed(0)


def test_calibration_and_test_partitions_are_rejected() -> None:
    assert require_b2_validation_partition("validation") == "validation"
    with pytest.raises(ValueError, match="frozen validation partition"):
        require_b2_validation_partition("test")
    with pytest.raises(ValueError, match="frozen validation partition"):
        require_b2_validation_partition("calibration")


def test_canonical_class_ordering_is_sorted_unique_train_labels() -> None:
    labels = _class_labels()
    shuffled = list(reversed(labels))
    records = [[f"text-{label}", label] for label in shuffled for _ in range(2)]
    assert in_scope_class_labels(records) == labels


def test_probability_columns_must_match_canonical_class_mapping() -> None:
    labels = _class_labels()
    probabilities = np.full((4, 150), 1.0 / 150.0, dtype=np.float64)
    verify_b2_probability_columns(probabilities, labels)
    with pytest.raises(ValueError, match="canonical class labels"):
        verify_b2_probability_columns(np.full((4, 149), 1.0 / 149.0), labels)
    with pytest.raises(ValueError, match="unique and in canonical sorted order"):
        verify_b2_probability_columns(probabilities, list(reversed(labels)))


def test_loaded_config_must_be_frozen_bert() -> None:
    verify_loaded_bert_config(_bert_config())
    with pytest.raises(ValueError, match="requires a BERT encoder"):
        verify_loaded_bert_config(_bert_config(model_type="gpt2"))
    with pytest.raises(ValueError, match="bidirectional"):
        verify_loaded_bert_config(_bert_config(is_decoder=True))
    with pytest.raises(ValueError, match="does not match the B2 freeze"):
        verify_loaded_bert_config(_bert_config(hidden_size=1024))


def test_load_frozen_model_requires_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_config(*args: object, **kwargs: object) -> SimpleNamespace:
        seen["config"] = kwargs
        return _bert_config()

    def fake_tokenizer(*args: object, **kwargs: object) -> object:
        seen["tokenizer"] = kwargs
        return object()

    def fake_model(*args: object, **kwargs: object) -> SimpleNamespace:
        seen["model"] = kwargs
        return SimpleNamespace(config=_bert_config())

    monkeypatch.setattr("temper.baselines.encoder.AutoConfig.from_pretrained", fake_config)
    monkeypatch.setattr("temper.baselines.encoder.AutoTokenizer.from_pretrained", fake_tokenizer)
    monkeypatch.setattr(
        "temper.baselines.encoder.AutoModelForSequenceClassification.from_pretrained",
        fake_model,
    )
    load_frozen_b2_model_and_tokenizer()
    assert seen["config"]["revision"] == B2_MODEL_REVISION
    assert seen["tokenizer"]["revision"] == B2_TOKENIZER_REVISION
    assert seen["model"]["revision"] == B2_MODEL_REVISION
    assert "main" not in seen["config"].get("revision", "")


def test_per_seed_artifact_names_cannot_collide() -> None:
    names = [b2_run_artifacts(seed) for seed in B2_SEEDS]
    run_ids = [item.run_id for item in names]
    results = [item.result_name for item in names]
    manifests = [item.manifest_name for item in names]
    assert len(set(run_ids)) == 3
    assert len(set(results)) == 3
    assert len(set(manifests)) == 3
    assert names[0].result_name == "EXP-0001-B2-validation-seed-13.npz"


def test_artifact_overwrite_is_refused(tmp_path: Path) -> None:
    labels = np.array([0, 1], dtype=np.int64)
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    metrics = BaselineMetrics(accuracy=1.0, macro_f1=1.0, nll=0.1, multiclass_brier=0.1)
    path = tmp_path / b2_run_artifacts(13).result_name
    write_prediction_artifact(
        path,
        labels=labels,
        predictions=labels,
        probabilities=probabilities,
        class_labels=np.array(["a", "b"]),
        metrics=metrics,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_prediction_artifact(
            path,
            labels=labels,
            predictions=labels,
            probabilities=probabilities,
            class_labels=np.array(["a", "b"]),
            metrics=metrics,
        )


def test_hardware_and_frozen_config_serialize_safely() -> None:
    snapshot = collect_b2_hardware_software()
    required = {
        "cpu",
        "gpu",
        "ram_gb",
        "device",
        "dtype",
        "batch_size",
        "max_length",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "python",
        "torch",
        "transformers",
        "numpy",
    }
    assert required <= set(snapshot)
    assert snapshot["gpu"] in {None, "none"} or isinstance(snapshot["gpu"], str)
    json.dumps(frozen_b2_hyperparameters())
    json.dumps({key: snapshot[key] for key in snapshot if snapshot[key] is not None})


def test_aggregate_reporting_does_not_select_best_seed() -> None:
    per_seed = {
        13: BaselineMetrics(accuracy=0.5, macro_f1=0.40, nll=1.5, multiclass_brier=0.6),
        21: BaselineMetrics(accuracy=0.9, macro_f1=0.80, nll=0.4, multiclass_brier=0.2),
        37: BaselineMetrics(accuracy=0.7, macro_f1=0.60, nll=0.9, multiclass_brier=0.4),
    }
    summary = aggregate_b2_metrics(per_seed)
    assert summary["seeds"] == [13, 21, 37]
    assert summary["mean_macro_f1"] == pytest.approx(0.6)
    assert summary["std_macro_f1"] == pytest.approx(float(np.std([0.4, 0.8, 0.6], ddof=1)))
    assert "best_seed" not in summary
    assert "max_macro_f1" not in summary
    with pytest.raises(ValueError, match="exactly seeds"):
        aggregate_b2_metrics({13: per_seed[13], 21: per_seed[21]})
    with pytest.raises(ValueError, match="exactly seeds"):
        aggregate_b2_metrics({**per_seed, 99: per_seed[13]})


class _FakeTokenizer:
    def __call__(
        self,
        texts: list[str],
        truncation: bool,
        padding: str,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert truncation is True
        assert padding == "max_length"
        assert max_length == 128
        count = len(texts)
        return {
            "input_ids": torch.ones(count, max_length, dtype=torch.long),
            "attention_mask": torch.ones(count, max_length, dtype=torch.long),
        }


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(128, 150)
        self.config = _bert_config()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        logits = self.classifier(input_ids.float())
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


def test_tiny_training_loop_keeps_final_epoch_weights() -> None:
    model = _FakeEncoder()
    texts = [f"example-{index}" for index in range(16)]
    labels = np.zeros(16, dtype=np.int64)
    info = train_frozen_b2(
        model=model,
        tokenizer=_FakeTokenizer(),
        texts=texts,
        labels=labels,
        seed=13,
        device=torch.device("cpu"),
    )
    assert info["epochs"] == 3
    assert info["amp_grad_scaler"] is False


def test_run_b2_cli_has_no_test_partition_flag(run_b2_module: ModuleType) -> None:
    parser_source = Path("experiments/EXP-0001/run_b2.py").read_text(encoding="utf-8")
    assert "--partition" not in parser_source
    assert 'choices=("validation", "test")' not in parser_source
    assert 'payload["test"]' not in parser_source
    assert "calibration_indices" not in parser_source


def test_run_b2_refuses_existing_manifest(run_b2_module: ModuleType, tmp_path: Path) -> None:
    names = b2_run_artifacts(13)
    manifest_path = tmp_path / "manifests" / names.manifest_name
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    dataset_path = tmp_path / "data_full.json"
    dataset_path.write_bytes(b"{}")
    arguments = argparse.Namespace(
        dataset=dataset_path,
        archive_sha256="a",
        canonical_sha256=hashlib.sha256(b"{}").hexdigest(),
        splits=tmp_path / "splits.json",
        output=tmp_path,
        seed=13,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite manifest"):
        run_b2_module._run_one_seed(arguments)


def test_split_validation_still_covers_exp0001_contract() -> None:
    labels = _class_labels()
    payload = {
        "train": [[f"train-{label}-{index}", label] for label in labels for index in range(100)],
        "val": [[f"val-{label}-{index}", label] for label in labels for index in range(20)],
        "test": [[f"test-{label}-{index}", label] for label in labels for index in range(30)],
        "oos_train": [[f"oos-train-{index}", "oos"] for index in range(100)],
        "oos_val": [[f"oos-val-{index}", "oos"] for index in range(100)],
        "oos_test": [[f"oos-test-{index}", "oos"] for index in range(1_000)],
    }
    validation_labels = [record[1] for record in payload["val"]]
    splits = freeze_clinc150_splits(validation_labels=validation_labels, seed=42)
    validate_exp0001_splits(splits, payload)
