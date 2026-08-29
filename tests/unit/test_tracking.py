from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wsi_lab.config.models import AppConfig, DatasetConfig, WorkspaceConfig
from wsi_lab.errors import TrackingError
from wsi_lab.tracking import TrackingRunSpec, create_tracking, numeric_metrics


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        dataset=DatasetConfig(root=tmp_path / "dataset"),
        workspace=WorkspaceConfig(root=tmp_path / "workspace"),
    )
    config.logging.tensorboard = False
    return config


class _Run:
    def __init__(self, *, fail_log: bool = False) -> None:
        self.fail_log = fail_log
        self.defined: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.logged: list[dict[str, float | int]] = []
        self.finish_codes: list[int] = []

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        self.defined.append((args, kwargs))

    def log(self, values: dict[str, float | int]) -> None:
        if self.fail_log:
            raise RuntimeError("logging unavailable")
        self.logged.append(values)

    def finish(self, *, exit_code: int) -> None:
        self.finish_codes.append(exit_code)


def _fake_wandb(monkeypatch: pytest.MonkeyPatch, run: _Run, captured: dict[str, Any]) -> None:
    def init(**kwargs: Any) -> _Run:
        captured.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))


def test_disabled_tracking_does_not_import_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", None)

    with create_tracking(
        config,
        tmp_path / "run",
        TrackingRunSpec(job_type="train-rl", fingerprint="abc"),
    ) as tracking:
        assert tracking.writer is None


def test_tensorboard_works_without_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch.utils.tensorboard as tensorboard_module

    config = _config(tmp_path)
    config.logging.tensorboard = True
    captured: dict[str, Any] = {"events": []}

    class Writer:
        def __init__(self, *, log_dir: str) -> None:
            captured["log_dir"] = log_dir

        def add_scalar(self, name: str, value: float, step: int) -> None:
            captured["events"].append((name, value, step))

        def flush(self) -> None:
            captured["flushed"] = True

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(tensorboard_module, "SummaryWriter", Writer)
    monkeypatch.setitem(sys.modules, "wandb", None)

    with create_tracking(
        config,
        tmp_path / "run",
        TrackingRunSpec(job_type="train-belief", fingerprint="abc"),
    ) as tracking:
        tracking.add_scalar("belief/epoch/train_loss", 0.25, 7)
        tracking.flush()

    assert captured == {
        "events": [("belief/epoch/train_loss", 0.25, 7)],
        "log_dir": str(tmp_path / "run" / "tensorboard"),
        "flushed": True,
        "closed": True,
    }


def test_online_wandb_uses_deterministic_identity_and_custom_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.logging.wandb = True
    config.logging.wandb_project = "pathology"
    config.logging.wandb_entity = "research-team"
    config.logging.wandb_tags = ("baseline", "camelyon16")
    run = _Run()
    captured: dict[str, Any] = {}
    _fake_wandb(monkeypatch, run, captured)
    fingerprint = "a" * 64

    with create_tracking(
        config,
        tmp_path / "run",
        TrackingRunSpec(
            job_type="train-rl",
            fingerprint=fingerprint,
            group=f"ppo-{fingerprint}",
        ),
    ) as tracking:
        assert tracking.writer is tracking
        tracking.add_scalars(
            {"ppo/policy_loss": 1.25, "training/learning_rate": 2.5e-4},
            128,
        )
        tracking.add_scalar("ppo/value_loss", 2.0, 64)

    assert captured["project"] == "pathology"
    assert captured["entity"] == "research-team"
    assert captured["id"] == f"train-rl-{fingerprint}"
    assert captured["resume"] == "allow"
    assert captured["force"] is True
    assert captured["mode"] == "online"
    assert captured["name"] == "camelyon16-train-rl-aaaaaaaa"
    assert captured["group"] == f"ppo-{fingerprint}"
    assert captured["job_type"] == "train-rl"
    assert captured["tags"] == ("baseline", "camelyon16")
    assert captured["config"]["_tracking"] == {
        "fingerprint": fingerprint,
        "job_type": "train-rl",
        "group": f"ppo-{fingerprint}",
    }
    assert run.logged == [
        {
            "global_step": 128,
            "ppo/policy_loss": 1.25,
            "training/learning_rate": 2.5e-4,
        },
        {"global_step": 64, "ppo/value_loss": 2.0},
    ]
    assert (("global_step",), {}) in run.defined
    assert (("ppo/*",), {"step_metric": "global_step"}) in run.defined
    assert (("training/*",), {"step_metric": "global_step"}) in run.defined
    assert run.finish_codes == [0]


def test_offline_wandb_uses_local_segment_without_resume_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.logging.wandb = True
    config.logging.wandb_mode = "offline"
    run = _Run()
    captured: dict[str, Any] = {}
    _fake_wandb(monkeypatch, run, captured)

    with create_tracking(
        config,
        tmp_path / "run",
        TrackingRunSpec(job_type="train-belief", fingerprint="belief-fingerprint"),
    ):
        pass

    assert captured["mode"] == "offline"
    assert captured["force"] is False
    assert "id" not in captured
    assert "resume" not in captured


def test_enabled_wandb_reports_missing_optional_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.logging.wandb = True
    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(TrackingError, match=r"install.*\[wandb\]"):
        create_tracking(
            config,
            tmp_path / "run",
            TrackingRunSpec(job_type="train-belief", fingerprint="abc"),
        )


def test_wandb_log_failure_fails_the_session_and_marks_the_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.logging.wandb = True
    run = _Run(fail_log=True)
    _fake_wandb(monkeypatch, run, {})

    with pytest.raises(TrackingError, match="log scalar metrics"), create_tracking(
        config,
        tmp_path / "run",
        TrackingRunSpec(job_type="train-rl", fingerprint="abc"),
    ) as tracking:
        tracking.add_scalar("ppo/loss", 1.0, 8)

    assert run.finish_codes == [1]


def test_numeric_metrics_flattens_mappings_and_excludes_booleans() -> None:
    assert dict(
        numeric_metrics(
            "validation",
            {"accuracy": 0.8, "counts": {"tp": 4}, "complete": True, "rows": [1, 2]},
        )
    ) == {"validation/accuracy": 0.8, "validation/counts/tp": 4.0}
