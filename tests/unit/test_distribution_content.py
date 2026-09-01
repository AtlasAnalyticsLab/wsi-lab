from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_FILES = (
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "environment.yml",
    "pyproject.toml",
)
PROJECT_DIRECTORIES = ("configs", "docs", "src")
USER_ASSETS = (
    "LICENSE",
    "README.md",
    "environment.yml",
    "configs/camelyon16.example.yaml",
    "configs/camelyon16_17.example.yaml",
    "configs/camelyon16.yaml",
    "configs/camelyon17.example.yaml",
    "configs/generic.example.yaml",
    "configs/tcga_nsclc.example.yaml",
    "docs/CONFIGURATION.md",
    "docs/DATA_FORMAT.md",
    "docs/DESIGN.md",
    "docs/INSTALLATION.md",
    "docs/THIRD_PARTY.md",
)


def test_wandb_is_an_explicit_optional_extra() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    environment = (PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8")

    assert 'wandb = ["wandb>=0.19"]' in pyproject
    assert "-e .[dev]" in environment
    assert "-e .[dev,wandb]" not in environment


def _copy_build_input(destination: Path) -> None:
    destination.mkdir()
    for relative in PROJECT_FILES:
        shutil.copy2(PROJECT_ROOT / relative, destination / relative)
    for relative in PROJECT_DIRECTORIES:
        shutil.copytree(PROJECT_ROOT / relative, destination / relative)


def _build_distributions(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    output = tmp_path / "dist"
    _copy_build_input(source)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(output),
        ],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    sdists = tuple(output.glob("*.tar.gz"))
    wheels = tuple(output.glob("*.whl"))
    assert len(sdists) == 1, sdists
    assert len(wheels) == 1, wheels
    return sdists[0], wheels[0]


def test_sdist_and_wheel_retain_user_facing_assets(tmp_path: Path) -> None:
    """Build in a temporary copy so generated files never touch the checkout."""

    sdist, wheel = _build_distributions(tmp_path)
    with tarfile.open(sdist, mode="r:gz") as archive:
        sdist_members = {PurePosixPath(name) for name in archive.getnames()}
    for relative in USER_ASSETS:
        expected = PurePosixPath(relative)
        assert any(member.parts[1:] == expected.parts for member in sdist_members), relative

    with zipfile.ZipFile(wheel) as archive:
        wheel_members = tuple(PurePosixPath(name) for name in archive.namelist())
        metadata_member = next(
            member
            for member in wheel_members
            if member.name == "METADATA" and member.parent.name.endswith(".dist-info")
        )
        metadata = archive.read(str(metadata_member)).decode("utf-8")
    assert PurePosixPath("wsi_lab/__init__.py") in wheel_members
    assert any(member.name == "entry_points.txt" for member in wheel_members)
    assert any(member.parts[-2:] == ("licenses", "LICENSE") for member in wheel_members)
    assert "License-Expression: Apache-2.0" in metadata
    assert "License-File: LICENSE" in metadata
    for relative in USER_ASSETS:
        suffix = PurePosixPath("share") / "wsi-lab" / relative
        assert any(
            member.parts[-len(suffix.parts) :] == suffix.parts for member in wheel_members
        ), relative
