from __future__ import annotations

from pathlib import Path

from wsi_lab.config.models import AppConfig, DatasetConfig, WorkspaceConfig
from wsi_lab.pipeline.runner import PipelineRunner
from wsi_lab.pipeline.stages import Stage


def test_pipeline_dry_run_calls_no_stage_and_writes_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config = AppConfig(
        dataset=DatasetConfig(root=tmp_path / "dataset"),
        workspace=WorkspaceConfig(root=workspace),
    )
    called = False

    def stage(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    runner = PipelineRunner(config)
    runner.register(Stage.SPLIT, stage)
    plan = runner.run((Stage.SPLIT,), dry_run=True)
    assert plan.stages
    assert not called
    assert not workspace.exists()
