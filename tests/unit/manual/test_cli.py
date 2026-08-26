from __future__ import annotations

from types import SimpleNamespace

import pytest

from wsi_lab import cli
from wsi_lab.config.models import AppConfig, DatasetConfig, WorkspaceConfig
from wsi_lab.errors import TestSplitAccessError


def _config(tmp_path) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(root=tmp_path / "dataset"),
        workspace=WorkspaceConfig(root=tmp_path / "workspace"),
    )


def test_manual_play_parser_exposes_split_slide_seed_and_test_guard() -> None:
    arguments = cli.build_parser().parse_args(
        (
            "manual-play",
            "--config",
            "config.yaml",
            "--split",
            "test",
            "--slide-id",
            "test_042",
            "--seed",
            "9",
            "--allow-test",
        )
    )

    assert arguments.command == "manual-play"
    assert arguments.split == "test"
    assert arguments.slide_id == "test_042"
    assert arguments.seed == 9
    assert arguments.allow_test is True


def test_manual_play_rejects_test_before_preprocessing_or_environment_access(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "_prepare_authorized_test_preprocessing",
        lambda *_: pytest.fail("test preprocessing ran before access guard"),
    )
    monkeypatch.setattr(
        cli,
        "_make_production_environment",
        lambda *_args, **_kwargs: pytest.fail("environment opened before access guard"),
    )

    with pytest.raises(TestSplitAccessError, match="--allow-test"):
        cli._manual_play_command(
            _config(tmp_path),
            "test",
            False,
            slide_id=None,
            seed=None,
        )


def test_manual_play_uses_production_environment_and_closes_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wsi_lab.manual.ui as manual_ui

    config = _config(tmp_path)
    config.rl.seed = 17
    environment = SimpleNamespace(close_calls=0)

    def close() -> None:
        environment.close_calls += 1

    environment.close = close
    split = SimpleNamespace(ids=lambda name: ("normal_001", "tumor_001"))
    captured = {}
    monkeypatch.setattr(cli, "_load_verified_split", lambda _: split)
    monkeypatch.setattr(cli, "_make_production_environment", lambda *_args, **_kwargs: environment)

    def run_manual_play(active_environment, **kwargs) -> None:
        captured.update(environment=active_environment, **kwargs)

    monkeypatch.setattr(manual_ui, "run_manual_play", run_manual_play)

    cli._manual_play_command(
        config,
        "val",
        False,
        slide_id="tumor_001",
        seed=None,
    )

    assert captured == {
        "environment": environment,
        "slide_ids": ("normal_001", "tumor_001"),
        "initial_slide_id": "tumor_001",
        "seed": 17,
    }
    assert environment.close_calls == 1
