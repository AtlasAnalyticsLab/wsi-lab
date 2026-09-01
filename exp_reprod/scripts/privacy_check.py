#!/usr/bin/env python3
"""Fail if the distributable package contains private or measured artifacts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

GENERATED_DIRECTORIES = {"analysis_outputs", "runs", "runtime", "work"}
FORBIDDEN_DIRECTORIES = {"__pycache__", "reference_results"}
FORBIDDEN_FILENAMES = {"machine.json"}
FORBIDDEN_SUFFIXES = {".h5", ".npy", ".npz", ".pdf", ".png", ".pt", ".pth", ".pyc"}
LOCAL_PATH_PATTERNS = (
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"/mnt/[^/\s]+/"),
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/Volumes/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
)
SECRET_PATTERNS = (
    re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|passwd)\s*[:=]\s*['\"][^'\"]+"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~-]+"),
)
FORBIDDEN_HISTORY_PATTERNS = (
    re.compile(r"(?:^|/)reference_results/", re.MULTILINE),
    re.compile(r"(?:^|/)machine\.json(?:$|\s)", re.MULTILINE),
    re.compile(r"(?:^|/)[^/\s]*commands?\.json(?:$|\s)", re.MULTILINE),
)


def main() -> int:
    findings: list[str] = []
    for directory, names, filenames in os.walk(PACKAGE_ROOT):
        current = Path(directory)
        for name in sorted(set(names) & GENERATED_DIRECTORIES):
            findings.append(f"generated directory is present: {current / name}")
        for name in sorted(set(names) & FORBIDDEN_DIRECTORIES):
            findings.append(f"forbidden result directory: {current / name}")
        names[:] = [
            name
            for name in names
            if name not in GENERATED_DIRECTORIES | FORBIDDEN_DIRECTORIES
        ]
        for filename in filenames:
            path = current / filename
            relative = path.relative_to(PACKAGE_ROOT)
            lower_name = filename.lower()
            if filename in FORBIDDEN_FILENAMES:
                findings.append(f"forbidden machine inventory: {relative}")
            if "command" in lower_name and path.suffix.lower() == ".json":
                findings.append(f"historical command manifest: {relative}")
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                findings.append(f"bundled binary/model/result artifact: {relative}")
            if path.resolve() == SELF:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for pattern in LOCAL_PATH_PATTERNS:
                if pattern.search(text):
                    findings.append(f"local absolute path in {relative}: {pattern.pattern}")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    findings.append(f"possible secret in {relative}: {pattern.pattern}")
            for pattern in FORBIDDEN_HISTORY_PATTERNS:
                if pattern.search(text):
                    findings.append(f"historical artifact reference in {relative}")

    if findings:
        print(json.dumps({"status": "failed", "findings": sorted(set(findings))}, indent=2))
        return 1
    print(json.dumps({"status": "clean", "package": PACKAGE_ROOT.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
