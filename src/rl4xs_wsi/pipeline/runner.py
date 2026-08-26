"""Shared idempotent stage runner used by every CLI command."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from rl4xs_wsi.config.models import AppConfig
from rl4xs_wsi.pipeline.planner import PipelinePlan, build_plan
from rl4xs_wsi.pipeline.stages import Stage, StageAction
from rl4xs_wsi.schemas import ArtifactManifest, ManifestStatus
from rl4xs_wsi.utils.atomic import atomic_json_dump
from rl4xs_wsi.utils.system import git_commit, system_metadata

LOGGER = logging.getLogger(__name__)
StageFunction = Callable[[AppConfig, Path, str, StageAction], dict[str, object] | None]


class PipelineRunner:
    """Execute a selected DAG using injected, testable stage implementations."""

    def __init__(self, config: AppConfig, *, project_root: str | Path | None = None) -> None:
        self.config = config
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self._functions: dict[Stage, StageFunction] = {}

    def register(self, stage: Stage, function: StageFunction) -> None:
        self._functions[stage] = function

    def plan(self, stages: Iterable[Stage] | None = None) -> PipelinePlan:
        return build_plan(self.config, stages)

    def run(self, stages: Iterable[Stage] | None = None, *, dry_run: bool = False) -> PipelinePlan:
        plan = self.plan(stages)
        if dry_run:
            return plan
        for item in plan.stages:
            if item.action is StageAction.SKIP:
                LOGGER.info("Skipping %s: %s", item.stage.value, item.reason)
                continue
            function = self._functions.get(item.stage)
            if function is None:
                raise RuntimeError(f"No implementation registered for stage {item.stage.value}")
            item.artifact.parent.mkdir(parents=True, exist_ok=True)
            incomplete = ArtifactManifest(
                artifact_type=item.stage.value,
                fingerprint=item.fingerprint,
                status=ManifestStatus.INCOMPLETE,
                git_commit=git_commit(self.project_root),
                metadata={"action": item.action.value, "system": system_metadata()},
            )
            atomic_json_dump(item.artifact, incomplete.model_dump(mode="json"))
            try:
                metadata = (
                    function(self.config, item.artifact.parent, item.fingerprint, item.action) or {}
                )
            except BaseException as exc:
                failed = incomplete.model_copy(
                    update={
                        "status": ManifestStatus.FAILED,
                        "metadata": {**incomplete.metadata, "error": repr(exc)},
                    }
                )
                atomic_json_dump(item.artifact, failed.model_dump(mode="json"))
                raise
            complete = incomplete.model_copy(
                update={
                    "status": ManifestStatus.COMPLETE,
                    "completed_at": datetime.now(UTC),
                    "metadata": {**incomplete.metadata, **metadata},
                }
            )
            atomic_json_dump(item.artifact, complete.model_dump(mode="json"))
        return plan
