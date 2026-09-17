"""Fail-closed hashing, atomic writes, and sidecar digests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Final

DIGEST_SCHEMA: Final = "temper.evidence.sidecar.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write bytes via a same-directory replace. Refuse overwrite."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def sidecar_digest_path(manifest_path: Path) -> Path:
    return Path(str(manifest_path) + ".sha256.json")


def write_manifest_sidecar(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    artifact_name: str,
    artifact_sha256: str,
) -> Path:
    path = sidecar_digest_path(manifest_path)
    payload = {
        "schema": DIGEST_SCHEMA,
        "manifest_name": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "artifact_name": artifact_name,
        "artifact_sha256": artifact_sha256,
    }
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest_sidecar(manifest_path: Path) -> dict[str, str]:
    path = sidecar_digest_path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"missing manifest digest sidecar: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != DIGEST_SCHEMA:
        raise ValueError(f"unsupported digest sidecar schema at {path}")
    required = ("manifest_name", "manifest_sha256", "artifact_name", "artifact_sha256")
    missing = [key for key in required if not isinstance(payload.get(key), str)]
    if missing:
        raise ValueError(f"digest sidecar missing string fields {missing}: {path}")
    return {key: str(payload[key]) for key in ("schema", *required)}


def verify_manifest_sidecar(
    *,
    manifest_path: Path,
    artifact_path: Path,
    expected_artifact_sha256: str,
) -> dict[str, str]:
    sidecar = read_manifest_sidecar(manifest_path)
    observed_manifest = sha256_file(manifest_path)
    observed_artifact = sha256_file(artifact_path)
    if sidecar["manifest_name"] != manifest_path.name:
        raise ValueError(
            "digest sidecar manifest_name does not match the manifest file: "
            f"{sidecar['manifest_name']} vs {manifest_path.name}"
        )
    if sidecar["artifact_name"] != artifact_path.name:
        raise ValueError(
            "digest sidecar artifact_name does not match the prediction artifact: "
            f"{sidecar['artifact_name']} vs {artifact_path.name}"
        )
    if sidecar["manifest_sha256"] != observed_manifest:
        raise ValueError(
            "manifest SHA-256 does not match digest sidecar: "
            f"sidecar {sidecar['manifest_sha256']}, observed {observed_manifest}"
        )
    if sidecar["artifact_sha256"] != observed_artifact:
        raise ValueError(
            "prediction artifact SHA-256 does not match digest sidecar: "
            f"sidecar {sidecar['artifact_sha256']}, observed {observed_artifact}"
        )
    if expected_artifact_sha256 != observed_artifact:
        raise ValueError(
            "prediction artifact SHA-256 does not match the manifest record: "
            f"manifest {expected_artifact_sha256}, observed {observed_artifact}"
        )
    return sidecar


def inspect_git_provenance() -> dict[str, object]:
    """Record HEAD and working-tree cleanliness. Do not invent a commit."""
    git = shutil.which("git")
    if git is None:
        return {
            "git_commit": None,
            "git_dirty": None,
            "git_status_porcelain": None,
        }
    try:
        commit = subprocess.check_output(  # nosec B603
            [git, "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        porcelain = subprocess.check_output(  # nosec B603
            [git, "status", "--porcelain"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        porcelain = None
    dirty = None if porcelain is None else bool(porcelain.strip())
    return {
        "git_commit": commit or None,
        "git_dirty": dirty,
        "git_status_porcelain": porcelain.strip() if porcelain and porcelain.strip() else None,
    }


def require_clean_git() -> str:
    """Fail closed when HEAD is unknown or the working tree is dirty."""
    snapshot = inspect_git_provenance()
    commit = snapshot["git_commit"]
    dirty = snapshot["git_dirty"]
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("unable to resolve git HEAD; refusing B2 execution")
    if dirty is not False:
        raise RuntimeError(
            "refusing B2 execution from a dirty or unreadable git working tree; "
            "commit or revert local changes so git_commit is unambiguous"
        )
    return commit
