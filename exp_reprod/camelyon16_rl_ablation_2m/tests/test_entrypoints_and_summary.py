"""Filesystem-safety contracts for public entry points and summary output."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from ablation import summarize as ablation_summarize

REPRODUCTION_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "candidate",
    (
        ablation_summarize.REPRODUCTION_DIR,
        ablation_summarize.REPRODUCTION_DIR / "generated-summary.json",
    ),
    ids=("package-root", "package-child"),
)
def test_summary_output_rejects_reproduction_package(candidate: Path) -> None:
    with pytest.raises(
        ablation_summarize.SummaryError,
        match="outside the committed reproduction package",
    ):
        ablation_summarize._safe_output_path(candidate)


@pytest.mark.parametrize("entrypoint", ("reproduce.py", "summarize.py"))
def test_public_entrypoints_disable_bytecode_before_importing_package(
    entrypoint: str,
    tmp_path: Path,
) -> None:
    probe = (
        "import runpy, sys; "
        "sys.dont_write_bytecode = False; "
        "runpy.run_path(sys.argv[1], run_name='entrypoint_probe'); "
        "print(sys.dont_write_bytecode)"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")

    completed = subprocess.run(
        [sys.executable, "-c", probe, str(REPRODUCTION_DIR / entrypoint)],
        cwd=tmp_path,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True"
