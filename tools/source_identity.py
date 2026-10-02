"""Verify source-file content hashes against a captured Git commit."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


def git_source_bytes(commit: str, path: str, *, cwd: Path | None = None) -> bytes:
    """Read a file's bytes from ``commit`` using Git's blob contents."""

    if not isinstance(commit, str) or not commit.strip():
        raise ValueError("commit must be non-empty text")
    if not isinstance(path, str) or not path or path.startswith(("/", "\\")):
        raise ValueError("path must be a repository-relative path")
    try:
        return subprocess.run(
            ["git", "cat-file", "blob", f"{commit}:{path}"],
            cwd=str(cwd or Path.cwd()),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot read Git source {commit}:{path}") from exc


def source_hashes(
    commit: str,
    paths: Sequence[str],
    *,
    cwd: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Return content SHA-256 values for repository files at ``commit``."""

    result: dict[str, dict[str, str]] = {}
    for path in paths:
        content = git_source_bytes(commit, path, cwd=cwd)
        result[path] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "basis": "git_blob_content",
            "commit": commit,
        }
    return result


def verify_source_hashes(
    commit: str,
    claims: Mapping[str, Mapping[str, object]],
    *,
    cwd: Path | None = None,
) -> list[str]:
    """Return explicit mismatch reasons; an empty list means verified."""

    errors: list[str] = []
    for path, claim in claims.items():
        expected = claim.get("sha256") if isinstance(claim, Mapping) else None
        try:
            actual = source_hashes(commit, [path], cwd=cwd)[path]["sha256"]
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if expected != actual:
            errors.append(f"source hash mismatch for {path}: expected {expected}, got {actual}")
    return errors


__all__ = ["git_source_bytes", "source_hashes", "verify_source_hashes"]
