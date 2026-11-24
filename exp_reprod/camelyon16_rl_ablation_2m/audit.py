#!/usr/bin/env python3
"""Read-only publication audit for privacy, output leakage, and package scope."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MAX_SOURCE_BYTES = 1_000_000
ALLOWED_FILES = {
    "README.md",
    "ablation/__init__.py",
    "audit.py",
    "base_config.yaml",
    "bin/atlaspatch-pinned",
    "environment.cpu.yml",
    "environment.yml",
    "reproduce.py",
    "study.json",
    "summarize.py",
    "ablation/analysis_cache.py",
    "ablation/contract.py",
    "ablation/launcher.py",
    "ablation/monitoring.py",
    "ablation/provenance.py",
    "ablation/rendering.py",
    "ablation/runner.py",
    "ablation/spec.py",
    "ablation/summarize.py",
    "ablation/telemetry.py",
    "ablation/variants.py",
    "ablation/verification.py",
    "tests/conftest.py",
    "tests/test_entrypoints_and_summary.py",
    "tests/test_publication_audit.py",
    "tests/test_runner_contract.py",
    "tests/test_spec_and_launcher.py",
    "tests/test_variants.py",
}
GENERATED_DIRECTORY_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "history",
    "logs",
    "metrics",
    "outputs",
    "results",
    "runs",
    "state",
    "summaries",
    "tensorboard",
    "wandb",
}
GENERATED_SUFFIXES = {
    ".ckpt",
    ".csv",
    ".db",
    ".err",
    ".h5",
    ".hdf5",
    ".jpg",
    ".jsonl",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".out",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".sqlite",
    ".sqlite3",
    ".svs",
    ".svg",
    ".tsv",
    ".tif",
    ".tiff",
}


def _forbidden_fragments() -> tuple[str, ...]:
    """Construct sensitive tokens without making the audit flag itself."""

    return (
        "/" + "home" + "/",
        "/" + "scratch" + "/",
        "/" + "Users" + "/",
        "/" + "gpfs" + "/",
        "/" + "lustre" + "/",
        "/" + "nfs" + "/",
        "/" + "private" + "/var/",
    )


def _walk_json_keys(value: Any, location: str = "$") -> list[str]:
    errors: list[str] = []
    denied = {
        "accuracy",
        "argv",
        "auc",
        "command",
        "commands",
        "cwd",
        "expected_artifacts",
        "invocation",
        "job_id",
        "metric",
        "metrics",
        "result",
        "results",
        "runtime_basis",
        "submission",
        "submitted_at",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in denied:
                errors.append(f"historical/output key {key!r} at {location}")
            errors.extend(_walk_json_keys(item, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_walk_json_keys(item, f"{location}[{index}]"))
    return errors


def audit() -> list[str]:
    errors: list[str] = []
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() or path.is_symlink())
    relative_files = {path.relative_to(ROOT).as_posix() for path in files}
    missing = sorted(ALLOWED_FILES - relative_files)
    unexpected = sorted(relative_files - ALLOWED_FILES)
    errors.extend(f"required file missing: {name}" for name in missing)
    errors.extend(f"file is not on the publication allowlist: {name}" for name in unexpected)

    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            errors.append(f"symlink is not publishable: {relative}")
            continue
        if any(part in GENERATED_DIRECTORY_NAMES for part in path.relative_to(ROOT).parts):
            errors.append(f"generated directory is present: {relative}")
        if path.suffix.casefold() in GENERATED_SUFFIXES:
            errors.append(f"generated/binary artifact is present: {relative}")
        if path.stat().st_size > MAX_SOURCE_BYTES:
            errors.append(f"unexpectedly large source file: {relative}")
        payload = path.read_bytes()
        if b"\0" in payload:
            errors.append(f"binary file is present: {relative}")
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 file is present: {relative}")
            continue
        lowered = text.casefold()
        for fragment in _forbidden_fragments():
            if fragment.casefold() in lowered:
                errors.append(f"machine-specific filesystem text in {relative}")
                break
        unc_prefix = chr(92) * 2
        has_unc_line = any(line.strip().startswith(unc_prefix) for line in text.splitlines())
        if re.search(r"(?i)\b[A-Z]:[\\/]", text) or has_unc_line:
            errors.append(f"machine-specific path syntax in {relative}")
        if re.search(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", text):
            errors.append(f"private key material in {relative}")
        if re.search(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
            r"client[_-]?secret|password|private[_-]?token)\b"
            r"\s*[:=]\s*['\"][^'\"]{8,}['\"]",
            text,
        ):
            errors.append(f"credential-like assignment in {relative}")
        if re.search(r"(?i)https?://[^\s/@:]+:[^\s/@]+@", text):
            errors.append(f"credential-bearing URL in {relative}")
        if re.search(
            r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            text,
        ):
            errors.append(f"email address in {relative}")
        if path.suffix == ".json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON in {relative}: {exc}")
            else:
                errors.extend(f"{relative}: {error}" for error in _walk_json_keys(value))

    repository = ROOT.parents[1]
    try:
        worktree = (
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=repository,
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )
    except OSError:
        worktree = False
    if worktree:
        for path in files:
            completed = subprocess.run(
                ["git", "check-ignore", "--no-index", "--quiet", str(path)],
                cwd=repository,
                check=False,
            )
            if completed.returncode == 0:
                errors.append(f"publication file is ignored by Git: {path.relative_to(ROOT)}")
            elif completed.returncode not in (1,):
                errors.append(f"could not inspect Git ignore state: {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    errors = audit()
    if errors:
        print("publication audit failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("publication audit passed: no private paths, prior outputs, or run history found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
