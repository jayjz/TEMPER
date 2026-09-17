"""Frozen EXP-0001 dataset identity. Not caller-controlled."""

from __future__ import annotations

from pathlib import Path

from temper.evidence import sha256_file

EXP0001_ARCHIVE_SHA256 = "0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2"
EXP0001_CANONICAL_SHA256 = "fb3217519e3c601c7a9b019dfd6744bed8f2564833e2b7eac4a406cacb462489"


def verify_exp0001_archive_claim(claimed_archive: str) -> str:
    """Reject a caller-supplied archive hash that is not the frozen identity."""
    if claimed_archive != EXP0001_ARCHIVE_SHA256:
        raise ValueError(
            "caller-supplied archive SHA-256 is not the frozen EXP-0001 identity: "
            f"expected {EXP0001_ARCHIVE_SHA256}, found {claimed_archive}"
        )
    return EXP0001_ARCHIVE_SHA256


def verify_exp0001_canonical_dataset(
    dataset_path: Path, *, claimed_canonical: str | None = None
) -> str:
    """Compare the dataset bytes to the frozen EXP-0001 canonical SHA-256.

    A caller-supplied hash is never authoritative. If one is provided, it must
    still equal the frozen identity; a modified file hashed to match itself is
    rejected.
    """
    expected = EXP0001_CANONICAL_SHA256
    if claimed_canonical is not None and claimed_canonical != expected:
        raise ValueError(
            "caller-supplied canonical SHA-256 is not the frozen EXP-0001 identity: "
            f"expected {expected}, found {claimed_canonical}"
        )
    observed = sha256_file(dataset_path)
    if observed != expected:
        raise ValueError(
            "dataset SHA-256 does not match frozen EXP-0001 canonical identity: "
            f"expected {expected}, observed {observed}"
        )
    return observed
