#!/usr/bin/env python3
"""Fail-fast setup verification; this never opens a WSI or runs an episode."""

from __future__ import print_function

import importlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream" / "HistoGym"
EXPECTED_COMMIT = "5b742760321d28a469d2ef9494e886a49d0aa7f7"
EXPECTED_PYTHON = "3.8.20"
EXPECTED_DEPENDENCIES = {
    "gym": "0.25.2",
    "numpy": "1.21.2",
    "Pillow": "9.5.0",
    "Shapely": "1.7.1",
    "matplotlib": "3.5.3",
    "opencv-python": "4.8.1.78",
    "openslide-python": "1.2.0",
    "psutil": "5.9.8",
}
EXPECTED_OPENSLIDE_LIBRARY = "3.4.1"


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM)] + list(args), text=True
    ).strip()


def version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    actual_commit = git("rev-parse", "HEAD")
    if actual_commit != EXPECTED_COMMIT:
        raise SystemExit(
            "HistoGym commit mismatch: expected %s, found %s"
            % (EXPECTED_COMMIT, actual_commit)
        )
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Pinned HistoGym checkout has tracked modifications")

    sys.path.insert(0, str(UPSTREAM))
    module = importlib.import_module("gymhisto")
    if pathlib.Path(module.__file__).resolve() != (UPSTREAM / "gymhisto.py").resolve():
        raise SystemExit("Imported gymhisto from the wrong checkout")

    import openslide

    dependencies = {
        name: version(name) for name in EXPECTED_DEPENDENCIES
    }
    mismatched = {
        name: {"expected": expected, "actual": dependencies.get(name)}
        for name, expected in EXPECTED_DEPENDENCIES.items()
        if dependencies.get(name) != expected
    }
    python_version = sys.version.split()[0]
    openslide_library = getattr(openslide, "__library_version__", None)
    if python_version != EXPECTED_PYTHON:
        mismatched["python"] = {
            "expected": EXPECTED_PYTHON,
            "actual": python_version,
        }
    if openslide_library != EXPECTED_OPENSLIDE_LIBRARY:
        mismatched["openslide_library"] = {
            "expected": EXPECTED_OPENSLIDE_LIBRARY,
            "actual": openslide_library,
        }
    if mismatched:
        raise SystemExit(
            "HistoGym environment does not match the pinned private specification: %s"
            % json.dumps(mismatched, sort_keys=True)
        )

    result = {
        "histogym_commit": actual_commit,
        "python": python_version,
        "conda_prefix": sys.prefix,
        "dependencies": dependencies,
        "openslide_library": openslide_library,
        "pinned_environment_verified": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
