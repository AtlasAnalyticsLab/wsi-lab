"""Path-independent source-tree verification for the recorded implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SourceCompatibilityError(RuntimeError):
    """The selected WSI-Lab source does not match the recorded implementation."""


class RuntimeCompatibilityError(RuntimeError):
    """The active software environment differs from the reproduction contract."""


def source_tree_digest(source_root: str | Path) -> tuple[str, int]:
    """Hash sorted Python sources using a documented sha256sum-like encoding."""

    root = Path(source_root).expanduser().resolve()
    package = root / "src" / "wsi_lab"
    if not package.is_dir():
        raise SourceCompatibilityError(f"WSI-Lab package directory is missing: {package}")
    paths = sorted(
        (path for path in package.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(root).as_posix()
        digest.update(f"{file_hash}  {relative}\n".encode())
    return digest.hexdigest(), len(paths)


def verify_source_tree(
    source_root: str | Path,
    *,
    expected_sha256: str,
    expected_files: int,
) -> None:
    """Verify source bytes without requiring Git history or a network fetch."""

    observed_sha256, observed_files = source_tree_digest(source_root)
    if observed_files != int(expected_files) or observed_sha256 != str(expected_sha256):
        raise SourceCompatibilityError(
            "src/wsi_lab does not match the recorded ablation implementation "
            f"(files={observed_files}, sha256={observed_sha256})"
        )


def _public_version(value: str) -> str:
    return str(value).split("+", maxsplit=1)[0]


def verify_runtime_environment(study: Mapping[str, Any]) -> None:
    """Require the portable reproduction versions and pinned AtlasPatch assets."""

    software = study.get("software_reproduction")
    atlaspatch = study.get("atlaspatch")
    if not isinstance(software, Mapping) or not isinstance(atlaspatch, Mapping):
        raise RuntimeCompatibilityError("study.json has no software reproduction contract")
    expected_python = str(software.get("python", ""))
    observed_python = ".".join(map(str, sys.version_info[:3]))
    mismatches: dict[str, tuple[str, str]] = {}
    if observed_python != expected_python:
        mismatches["python"] = (observed_python, expected_python)
    distributions = software.get("distributions")
    if not isinstance(distributions, Mapping):
        raise RuntimeCompatibilityError("study.json has no distribution version contract")
    for name, expected in distributions.items():
        try:
            observed = _public_version(importlib.metadata.version(str(name)))
        except importlib.metadata.PackageNotFoundError:
            observed = "missing"
        if observed != str(expected):
            mismatches[str(name)] = (observed, str(expected))
    if mismatches:
        rendered = ", ".join(
            f"{name}={observed!r} (expected {expected!r})"
            for name, (observed, expected) in sorted(mismatches.items())
        )
        raise RuntimeCompatibilityError(f"runtime version mismatch: {rendered}")

    try:
        sam_distribution = importlib.metadata.distribution("SAM-2")
        direct_url = json.loads(sam_distribution.read_text("direct_url.json") or "{}")
        vcs = direct_url.get("vcs_info", {})
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeCompatibilityError("SAM-2 VCS provenance is unavailable") from exc
    expected_commit = str(atlaspatch.get("sam2_commit"))
    if vcs.get("commit_id") != expected_commit:
        raise RuntimeCompatibilityError("SAM-2 does not match the recorded commit")
    if vcs.get("requested_revision") not in {None, expected_commit}:
        raise RuntimeCompatibilityError(
            "SAM-2 requested revision conflicts with the recorded commit"
        )
    try:
        from atlas_patch.core.config import default_sam2_config_path

        config_path = Path(default_sam2_config_path()).expanduser().resolve()
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except (ImportError, OSError) as exc:
        raise RuntimeCompatibilityError("AtlasPatch SAM2 configuration is unavailable") from exc
    if config_path.name != str(atlaspatch.get("sam2_config_filename")) or config_hash != str(
        atlaspatch.get("sam2_config_sha256")
    ):
        raise RuntimeCompatibilityError("AtlasPatch SAM2 configuration checksum mismatch")


__all__ = [
    "RuntimeCompatibilityError",
    "SourceCompatibilityError",
    "source_tree_digest",
    "verify_runtime_environment",
    "verify_source_tree",
]
