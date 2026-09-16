"""Explicit acquisition and validation for the CLINC150 full dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CLINC150_SOURCE_URL = "https://archive.ics.uci.edu/static/public/570/clinc150.zip"
_PARTITIONS = ("train", "val", "test", "oos_train", "oos_val", "oos_test")
_IN_SCOPE_COUNTS = {"train": 15_000, "val": 3_000, "test": 4_500}
_OOS_COUNTS = {"oos_train": 100, "oos_val": 100, "oos_test": 1_000}


@dataclass(frozen=True)
class Clinc150Dataset:
    """Validated CLINC150 metadata; raw contents remain outside version control."""

    path: Path
    source_url: str
    archive_sha256: str
    canonical_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_clinc150_payload(payload: object) -> dict[str, list[list[str]]]:
    """Validate the documented full-dataset schema without coercing records."""
    if not isinstance(payload, dict) or set(payload) != set(_PARTITIONS):
        raise ValueError(f"CLINC150 partitions must be exactly {_PARTITIONS}")

    validated: dict[str, list[list[str]]] = {}
    for partition in _PARTITIONS:
        records = payload[partition]
        if not isinstance(records, list):
            raise ValueError(f"CLINC150 partition {partition!r} must be a list")
        expected = _IN_SCOPE_COUNTS.get(partition, _OOS_COUNTS.get(partition))
        if len(records) != expected:
            raise ValueError(
                f"CLINC150 partition {partition!r} has {len(records)}, expected {expected}"
            )
        for record in records:
            if (
                not isinstance(record, list)
                or len(record) != 2
                or not all(isinstance(value, str) and value for value in record)
            ):
                raise ValueError(f"CLINC150 partition {partition!r} has an invalid record")
        validated[partition] = records

    for partition, expected_per_label in (("train", 100), ("val", 20), ("test", 30)):
        labels = [record[1] for record in validated[partition]]
        unique_labels = set(labels)
        if len(unique_labels) != 150 or any(
            labels.count(label) != expected_per_label for label in unique_labels
        ):
            raise ValueError(f"CLINC150 {partition!r} must contain 150 balanced in-scope labels")
    return validated


def _canonical_payload_bytes(payload: dict[str, list[list[str]]]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _try_validate_archive_member(
    archive: zipfile.ZipFile, name: str
) -> dict[str, list[list[str]]] | None:
    """Return a schema-valid payload or None for an invalid lookalike member."""
    try:
        payload: Any = json.loads(archive.read(name).decode("utf-8"))
        return validate_clinc150_payload(payload)
    except (UnicodeDecodeError, TypeError, ValueError):
        return None


def _select_validated_clinc150_payload(archive: zipfile.ZipFile) -> dict[str, list[list[str]]]:
    """Select the unique schema-valid CLINC150 payload from an archive.

    Every member whose name ends with ``data_full.json`` is inspected. Invalid
    lookalikes are rejected. Identical valid copies collapse to one canonical
    payload. Distinct valid payloads fail closed. Selection never uses
    first-match wins and does not depend on ZIP member order.
    """
    candidate_names = [name for name in archive.namelist() if name.endswith("data_full.json")]
    if not candidate_names:
        raise ValueError("CLINC150 archive must contain a data_full.json member")

    unique_payloads: dict[str, dict[str, list[list[str]]]] = {}
    for name in candidate_names:
        payload = _try_validate_archive_member(archive, name)
        if payload is None:
            continue
        digest = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
        unique_payloads[digest] = payload

    if not unique_payloads:
        raise ValueError("CLINC150 archive contains no schema-valid data_full.json")
    if len(unique_payloads) > 1:
        raise ValueError(
            "CLINC150 archive contains multiple distinct valid data_full.json payloads"
        )
    return next(iter(unique_payloads.values()))


def acquire_clinc150(
    destination: Path, *, source_url: str = CLINC150_SOURCE_URL
) -> Clinc150Dataset:
    """Download UCI's canonical archive, validate it, and retain its SHA-256."""
    if urlparse(source_url).scheme != "https":
        raise ValueError("CLINC150 source URL must use HTTPS")
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "clinc150.zip"
    dataset_path = destination / "data_full.json"
    if not archive_path.exists():
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with (
                urllib.request.urlopen(source_url) as response,  # nosec B310: HTTPS is enforced above.
                temporary_path.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            temporary_path.replace(archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            validated_payload = _select_validated_clinc150_payload(archive)
    except zipfile.BadZipFile as error:
        raise ValueError("CLINC150 source archive is not a valid zip file") from error
    normalized_bytes = _canonical_payload_bytes(validated_payload)
    dataset_path.write_bytes(normalized_bytes)
    return Clinc150Dataset(
        path=dataset_path,
        source_url=source_url,
        archive_sha256=_sha256(archive_path),
        canonical_sha256=_sha256(dataset_path),
    )
