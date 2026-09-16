"""Acquire CLINC150 and create immutable EXP-0001 split metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temper.datasets import acquire_clinc150, freeze_clinc150_splits, validate_clinc150_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/EXP-0001"))
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    dataset = acquire_clinc150(arguments.root / "data")
    split_path = arguments.root / "splits" / f"clinc150-full-seed-{arguments.seed}.json"
    payload = validate_clinc150_payload(json.loads(dataset.path.read_text(encoding="utf-8")))
    splits = freeze_clinc150_splits(
        validation_labels=[record[1] for record in payload["val"]], seed=arguments.seed
    )
    splits.write(split_path)
    print(f"dataset={dataset.path}")
    print(f"archive_sha256={dataset.archive_sha256}")
    print(f"canonical_sha256={dataset.canonical_sha256}")
    print(f"splits={split_path}")


if __name__ == "__main__":
    main()
