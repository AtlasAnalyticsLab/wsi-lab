#!/usr/bin/env python3
"""Verify the executable WSI-Lab source snapshot without requiring Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY_ROOT = Path(
    os.environ.get("WSI_LAB_SOURCE_ROOT", PACKAGE_ROOT.parent / "wsi-lab")
).expanduser().resolve()
DEFAULT_MANIFEST = PACKAGE_ROOT / "provenance/wsi_lab_source.sha256"
EXPECTED_MANIFEST_SHA256 = "ae742e4ccdfc0d8e951e8ace1a6bd67c3317cce5cd4100a8496d61a7ca7d32f4"
MEASURED_REVISION = "0deb7837364add790f9530c41f0ef0b5e616c05c"
MANIFEST_LINE = re.compile(r"^(?P<digest>[0-9a-f]{64})  (?P<path>[^\r\n]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_source_paths(repository_root: Path) -> set[str]:
    source_root = repository_root / "src/wsi_lab"
    if not source_root.is_dir():
        raise FileNotFoundError(f"WSI-Lab package source is missing: {source_root}")
    symlinks = sorted(path for path in source_root.rglob("*") if path.is_symlink())
    if symlinks:
        relative = [str(path.relative_to(repository_root)) for path in symlinks]
        raise RuntimeError(f"source snapshot contains unsupported symlinks: {relative}")
    paths = {
        path.relative_to(repository_root).as_posix()
        for path in source_root.rglob("*.py")
        if path.is_file()
    }
    paths.add("pyproject.toml")
    return paths


def load_manifest(manifest_path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid source manifest line {line_number}: {line!r}")
        relative = match.group("path")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe source manifest path: {relative!r}")
        if relative in records:
            raise ValueError(f"duplicate source manifest path: {relative!r}")
        records[relative] = match.group("digest")
    return records


def verify_source(repository_root: Path, manifest_path: Path) -> dict[str, object]:
    repository_root = repository_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "WSI-Lab source manifest identity differs from the submitted snapshot: "
            f"{manifest_sha256} != {EXPECTED_MANIFEST_SHA256}"
        )

    records = load_manifest(manifest_path)
    expected_paths = expected_source_paths(repository_root)
    recorded_paths = set(records)
    missing_records = sorted(expected_paths - recorded_paths)
    unexpected_records = sorted(recorded_paths - expected_paths)
    if missing_records or unexpected_records:
        raise RuntimeError(
            "WSI-Lab source file set differs from the submitted snapshot: "
            f"missing records={missing_records}, unexpected records={unexpected_records}"
        )

    mismatches: list[str] = []
    for relative in sorted(records):
        path = repository_root / relative
        if not path.is_file() or path.is_symlink():
            mismatches.append(f"{relative}: missing or not a regular file")
            continue
        observed = sha256_file(path)
        if observed != records[relative]:
            mismatches.append(f"{relative}: {observed} != {records[relative]}")
    if mismatches:
        raise RuntimeError(
            "WSI-Lab source content differs from the submitted snapshot: "
            + "; ".join(mismatches)
        )

    return {
        "status": "verified",
        "identity_scheme": "sha256-file-manifest-v1",
        "manifest": manifest_path.relative_to(PACKAGE_ROOT).as_posix(),
        "manifest_sha256": manifest_sha256,
        "file_count": len(records),
        "measured_revision": MEASURED_REVISION,
        "git_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_source(args.repository_root, args.manifest),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
