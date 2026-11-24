"""The publication audit itself is part of the test contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def audit_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "audit.py"
    specification = importlib.util.spec_from_file_location("ablation_publication_audit", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_publication_tree(
    audit_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Build an exact benign tree that is removed even when a test fails."""

    with tempfile.TemporaryDirectory(prefix="wsi-lab-publication-audit-") as directory:
        root = Path(directory)
        for relative in audit_module.ALLOWED_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = "{}\n" if path.suffix == ".json" else "benign publication source\n"
            path.write_text(content, encoding="utf-8")
        monkeypatch.setattr(audit_module, "ROOT", root)
        monkeypatch.setattr(
            audit_module.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
        )
        yield root


def test_publication_audit_passes(audit_module: ModuleType) -> None:
    assert audit_module.audit() == []


def test_benign_exact_publication_tree_passes(
    audit_module: ModuleType,
    isolated_publication_tree: Path,
) -> None:
    assert isolated_publication_tree.is_dir()
    assert audit_module.audit() == []


def test_unexpected_result_file_and_directory_are_rejected(
    audit_module: ModuleType,
    isolated_publication_tree: Path,
) -> None:
    leaked = isolated_publication_tree / "results" / "metrics.csv"
    leaked.parent.mkdir()
    leaked.write_text("synthetic-placeholder\n", encoding="utf-8")

    errors = audit_module.audit()

    assert any(
        "file is not on the publication allowlist: results/metrics.csv" in error for error in errors
    )
    assert any("generated directory is present: results/metrics.csv" in error for error in errors)
    assert any(
        "generated/binary artifact is present: results/metrics.csv" in error for error in errors
    )


def test_hidden_secret_style_file_is_rejected(
    audit_module: ModuleType,
    isolated_publication_tree: Path,
) -> None:
    hidden = isolated_publication_tree / ".env"
    credential_name = "api" + "_" + "key"
    hidden.write_text(
        f'{credential_name} = "synthetic-placeholder-value"\n',
        encoding="utf-8",
    )

    errors = audit_module.audit()

    assert any("file is not on the publication allowlist: .env" in error for error in errors)
    assert any("credential-like assignment in .env" in error for error in errors)


def test_forbidden_json_result_and_history_keys_are_rejected(
    audit_module: ModuleType,
    isolated_publication_tree: Path,
) -> None:
    (isolated_publication_tree / "study.json").write_text(
        json.dumps(
            {
                "nested": {
                    "result": {"value": "synthetic-placeholder"},
                    "commands": ["historical invocation"],
                }
            }
        ),
        encoding="utf-8",
    )

    errors = audit_module.audit()

    assert any("historical/output key 'result'" in error for error in errors)
    assert any("historical/output key 'commands'" in error for error in errors)
