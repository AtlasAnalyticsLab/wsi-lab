"""Persistent validation-model selection for recurrent PPO training."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from wsi_lab.rl.checkpoints import inspect_ppo_checkpoint, save_ppo_checkpoint
from wsi_lab.utils.atomic import atomic_json_dump

BEST_CHECKPOINT_FILENAME = "best_validation.pt"
BEST_METADATA_FILENAME = "best_validation.json"
BEST_ARCHIVE_DIRECTORY = "best_validation_archive"
BEST_METADATA_SCHEMA_VERSION = 1

SelectionMetric = Literal["auroc", "accuracy"]


def _optional_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"validation {name} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        return None
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"validation {name} must lie in [0,1], got {result}")
    return result


@dataclass(frozen=True)
class BestValidationRecord:
    """The validation result that owns the dedicated best checkpoint."""

    selection_metric: SelectionMetric
    selection_value: float
    global_step: int
    update: int
    checkpoint: str
    auroc: float | None
    accuracy: float | None
    metrics: Mapping[str, Any]
    dependencies: Mapping[str, str]

    @property
    def metric_name(self) -> SelectionMetric:
        """Public alias used by logging/reporting callers."""

        return self.selection_metric

    @property
    def metric_value(self) -> float:
        """Public alias used by logging/reporting callers."""

        return self.selection_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BEST_METADATA_SCHEMA_VERSION,
            "metric_name": self.selection_metric,
            "metric_value": self.selection_value,
            "selection_metric": self.selection_metric,
            "selection_value": self.selection_value,
            "global_step": self.global_step,
            "update": self.update,
            "checkpoint": self.checkpoint,
            "auroc": self.auroc,
            "accuracy": self.accuracy,
            "metrics": dict(self.metrics),
            "dependencies": dict(self.dependencies),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BestValidationRecord:
        if value.get("schema_version") != BEST_METADATA_SCHEMA_VERSION:
            raise ValueError("unsupported best-validation metadata schema")
        selection_metric = value.get("selection_metric")
        if selection_metric not in ("auroc", "accuracy"):
            raise ValueError("best-validation metadata has an invalid selection metric")
        selection_value = value.get("selection_value")
        if (
            isinstance(selection_value, bool)
            or not isinstance(selection_value, Real)
            or not math.isfinite(float(selection_value))
            or not 0.0 <= float(selection_value) <= 1.0
        ):
            raise ValueError("best-validation metadata has an invalid selection value")
        counters: dict[str, int] = {}
        for name in ("global_step", "update"):
            counter = value.get(name)
            if isinstance(counter, bool) or not isinstance(counter, Integral) or int(counter) < 0:
                raise ValueError(f"best-validation metadata has an invalid {name}")
            counters[name] = int(counter)
        checkpoint = value.get("checkpoint")
        if checkpoint != BEST_CHECKPOINT_FILENAME:
            raise ValueError("best-validation metadata references an unexpected checkpoint")
        metrics = value.get("metrics")
        dependencies = value.get("dependencies")
        if not isinstance(metrics, Mapping) or not isinstance(dependencies, Mapping):
            raise ValueError("best-validation metadata has invalid mapping fields")
        auroc = _optional_metric(value, "auroc")
        accuracy = _optional_metric(value, "accuracy")
        if selection_metric == "auroc" and auroc != float(selection_value):
            raise ValueError("best-validation AUROC does not match its selection value")
        if selection_metric == "accuracy" and accuracy != float(selection_value):
            raise ValueError("best-validation accuracy does not match its selection value")
        return cls(
            selection_metric=selection_metric,
            selection_value=float(selection_value),
            global_step=counters["global_step"],
            update=counters["update"],
            checkpoint=checkpoint,
            auroc=auroc,
            accuracy=accuracy,
            metrics=dict(metrics),
            dependencies={str(key): str(item) for key, item in dependencies.items()},
        )


@dataclass(frozen=True)
class BestCheckpointDecision:
    """Result of considering one validation pass for model selection."""

    improved: bool
    best: BestValidationRecord
    checkpoint_path: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "improved": self.improved,
            "best": self.best.to_dict(),
            "checkpoint_path": str(self.checkpoint_path),
            "metadata_path": str(self.metadata_path),
        }


def _candidate_record(
    metrics: Mapping[str, Any],
    *,
    global_step: int,
    update: int,
    dependencies: Mapping[str, str],
) -> BestValidationRecord:
    if global_step < 0 or update < 0:
        raise ValueError("global_step and update must be non-negative")
    auroc = _optional_metric(metrics, "auroc")
    accuracy = _optional_metric(metrics, "accuracy")
    if auroc is not None:
        selection_metric: SelectionMetric = "auroc"
        selection_value = auroc
    elif accuracy is not None:
        selection_metric = "accuracy"
        selection_value = accuracy
    else:
        raise ValueError("validation metrics must contain a finite AUROC or accuracy")
    return BestValidationRecord(
        selection_metric=selection_metric,
        selection_value=selection_value,
        global_step=int(global_step),
        update=int(update),
        checkpoint=BEST_CHECKPOINT_FILENAME,
        auroc=auroc,
        accuracy=accuracy,
        metrics=dict(metrics),
        dependencies={str(key): str(value) for key, value in dependencies.items()},
    )


def _is_improvement(candidate: BestValidationRecord, previous: BestValidationRecord) -> bool:
    if candidate.selection_metric != previous.selection_metric:
        # AUROC is the primary selection objective. Accuracy is comparable only
        # while no finite validation AUROC has been observed.
        return candidate.selection_metric == "auroc"
    # Exact ties deliberately retain the earlier checkpoint. This makes model
    # selection deterministic across uninterrupted and resumed runs.
    return candidate.selection_value > previous.selection_value


class BestValidationCheckpointTracker:
    """Select and atomically persist the best validation PPO checkpoint.

    The selection record is embedded in the checkpoint and mirrored to JSON.
    On construction the checkpoint is authoritative, allowing the JSON sidecar
    to be repaired if a process stops between the two atomic replacements.
    """

    def __init__(
        self,
        checkpoint_directory: str | Path,
        *,
        resume_existing: bool = True,
    ) -> None:
        self.directory = Path(checkpoint_directory).expanduser().resolve()
        self.checkpoint_path = self.directory / BEST_CHECKPOINT_FILENAME
        self.metadata_path = self.directory / BEST_METADATA_FILENAME
        self.quarantine_directory = (
            None if resume_existing else self._quarantine_existing_selection()
        )
        self._best = self._load_best()

    def _quarantine_existing_selection(self) -> Path | None:
        """Move selection artifacts aside before an explicitly fresh run.

        The archive is nested below the checkpoint directory and neither its
        directory nor its contained filenames match ``checkpoint_*.pt``.
        Consequently it cannot become a trainer or planner resume candidate.
        """

        sources = (self.checkpoint_path, self.metadata_path)
        if not any(source.exists() or source.is_symlink() for source in sources):
            return None
        archive_root = self.directory / BEST_ARCHIVE_DIRECTORY
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_directory = Path(tempfile.mkdtemp(prefix="stale_", dir=archive_root))
        moved: list[tuple[Path, Path]] = []
        try:
            for source in sources:
                if not (source.exists() or source.is_symlink()):
                    continue
                destination = archive_directory / source.name
                os.replace(source, destination)
                moved.append((source, destination))
        except BaseException:
            # Best-effort rollback keeps a failed quarantine from silently
            # discarding the only visible selection state.
            for source, destination in reversed(moved):
                if not (source.exists() or source.is_symlink()) and (
                    destination.exists() or destination.is_symlink()
                ):
                    with suppress(OSError):
                        os.replace(destination, source)
            with suppress(OSError):
                archive_directory.rmdir()
            raise
        return archive_directory

    @property
    def best(self) -> BestValidationRecord | None:
        return self._best

    def _load_best(self) -> BestValidationRecord | None:
        if not self.checkpoint_path.exists():
            if self.metadata_path.exists():
                raise ValueError(
                    f"best-validation metadata exists without its checkpoint: {self.metadata_path}"
                )
            return None
        loaded = inspect_ppo_checkpoint(self.checkpoint_path)
        embedded = loaded.extra.get("best_validation")
        if not isinstance(embedded, Mapping):
            raise ValueError(
                f"best-validation checkpoint lacks selection metadata: {self.checkpoint_path}"
            )
        record = BestValidationRecord.from_dict(embedded)
        if record.global_step != loaded.global_step or record.update != loaded.update:
            raise ValueError("best-validation counters do not match the checkpoint payload")

        persisted: BestValidationRecord | None = None
        if self.metadata_path.exists():
            try:
                raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                raw = None
            if isinstance(raw, Mapping):
                try:
                    persisted = BestValidationRecord.from_dict(raw)
                except ValueError:
                    persisted = None
        # The checkpoint is written first and embeds the record. Rewrite a
        # missing, damaged, or stale sidecar after an interrupted update.
        if persisted != record:
            atomic_json_dump(self.metadata_path, record.to_dict())
        return record

    def consider(
        self,
        metrics: Mapping[str, Any],
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,
        global_step: int,
        update: int,
        scheduler: Any | None = None,
        dependencies: Mapping[str, str] | None = None,
    ) -> BestCheckpointDecision:
        """Save the current training state when ``metrics`` establish a new best."""

        current_dependencies = {str(key): str(value) for key, value in (dependencies or {}).items()}
        candidate = _candidate_record(
            metrics,
            global_step=global_step,
            update=update,
            dependencies=current_dependencies,
        )
        if self._best is not None and not _is_improvement(candidate, self._best):
            return BestCheckpointDecision(
                improved=False,
                best=self._best,
                checkpoint_path=self.checkpoint_path,
                metadata_path=self.metadata_path,
            )
        save_ppo_checkpoint(
            self.checkpoint_path,
            model,
            optimizer,
            config,
            global_step=global_step,
            update=update,
            scheduler=scheduler,
            dependencies=current_dependencies,
            extra={"best_validation": candidate.to_dict()},
        )
        atomic_json_dump(self.metadata_path, candidate.to_dict())
        self._best = candidate
        return BestCheckpointDecision(
            improved=True,
            best=candidate,
            checkpoint_path=self.checkpoint_path,
            metadata_path=self.metadata_path,
        )


__all__ = [
    "BEST_ARCHIVE_DIRECTORY",
    "BEST_CHECKPOINT_FILENAME",
    "BEST_METADATA_FILENAME",
    "BestCheckpointDecision",
    "BestValidationCheckpointTracker",
    "BestValidationRecord",
]
