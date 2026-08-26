from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_FILES = (
    "MANIFEST.in",
    "README.md",
    "environment.yml",
    "pyproject.toml",
)
PROJECT_DIRECTORIES = ("configs", "docs", "src")
USER_ASSETS = (
    "README.md",
    "environment.yml",
    "configs/camelyon16.example.yaml",
    "configs/camelyon16.yaml",
    "docs/CONFIGURATION.md",
    "docs/DATA_FORMAT.md",
    "docs/DESIGN.md",
    "docs/INSTALLATION.md",
    "docs/THIRD_PARTY.md",
)


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
    assert PurePosixPath("rl4xs_wsi/__init__.py") in wheel_members
    assert any(member.name == "entry_points.txt" for member in wheel_members)
    for relative in USER_ASSETS:
        suffix = PurePosixPath("share") / "rl4xs-wsi" / relative
        assert any(
            member.parts[-len(suffix.parts) :] == suffix.parts for member in wheel_members
        ), relative
