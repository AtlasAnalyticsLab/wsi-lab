"""Optional, stage-scoped TensorBoard and Weights & Biases scalar tracking."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from wsi_lab.config.models import AppConfig
from wsi_lab.errors import TrackingError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrackingRunSpec:
    """Stable metadata used to identify one tracked pipeline operation."""

    job_type: str
    fingerprint: str
    group: str | None = None


def numeric_metrics(prefix: str, value: Any) -> Iterator[tuple[str, float]]:
    """Yield flattened numeric leaves from a nested mapping."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            nested = f"{prefix}/{key}" if prefix else str(key)
            yield from numeric_metrics(nested, item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


class ScalarTracking:
    """Fan scalar events out to independently enabled tracking backends."""

    def __init__(self, tensorboard_writer: Any | None, wandb_run: Any | None) -> None:
        self._tensorboard_writer = tensorboard_writer
        self._wandb_run = wandb_run
        self._defined_wandb_namespaces: set[str] = set()
        self._closed = False
        if self._wandb_run is not None:
            self._call_wandb(
                "define the global step metric",
                self._wandb_run.define_metric,
                "global_step",
            )

    @property
    def writer(self) -> ScalarTracking | None:
        """Return a TensorBoard-compatible writer only when a backend is active."""

        if self._tensorboard_writer is None and self._wandb_run is None:
            return None
        return self

    @staticmethod
    def _call_wandb(operation: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            raise TrackingError(f"W&B could not {operation}: {exc}") from exc

    def _define_wandb_namespace(self, name: str) -> None:
        if self._wandb_run is None:
            return
        namespace = name.rsplit("/", 1)[0] if "/" in name else name
        if namespace in self._defined_wandb_namespaces:
            return
        pattern = f"{namespace}/*" if "/" in name else name
        self._call_wandb(
            f"define metric namespace {pattern!r}",
            self._wandb_run.define_metric,
            pattern,
            step_metric="global_step",
        )
        self._defined_wandb_namespaces.add(namespace)

    def add_scalars(self, metrics: Mapping[str, int | float], step: int) -> None:
        """Record one logical scalar event at an application-owned global step."""

        if self._closed:
            raise TrackingError("cannot log metrics after the tracking session has closed")
        values = {str(name): float(value) for name, value in metrics.items()}
        if not values:
            return
        if self._tensorboard_writer is not None:
            for name, value in values.items():
                self._tensorboard_writer.add_scalar(name, value, int(step))
        if self._wandb_run is not None:
            for name in values:
                self._define_wandb_namespace(name)
            self._call_wandb(
                "log scalar metrics",
                self._wandb_run.log,
                {"global_step": int(step), **values},
            )

    def add_scalar(self, name: str, value: int | float, step: int) -> None:
        """Implement the SummaryWriter scalar interface used by PPOTrainer."""

        self.add_scalars({name: value}, step)

    def log_mapping(self, prefix: str, payload: Mapping[str, Any], step: int) -> None:
        self.add_scalars(dict(numeric_metrics(prefix, payload)), step)

    def flush(self) -> None:
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.flush()

    def close(self, *, exit_code: int = 0) -> None:
        if self._closed:
            return
        self._closed = True
        tensorboard_writer = self._tensorboard_writer
        wandb_run = self._wandb_run
        self._tensorboard_writer = None
        self._wandb_run = None
        first_error: BaseException | None = None
        if tensorboard_writer is not None:
            try:
                tensorboard_writer.close()
            except BaseException as exc:
                first_error = exc
        if wandb_run is not None:
            try:
                self._call_wandb(
                    "finish the run",
                    wandb_run.finish,
                    exit_code=int(exit_code),
                )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> ScalarTracking:
        return self

    def __exit__(
        self, exc_type: Any, exc: BaseException | None, traceback: Any
    ) -> Literal[False]:
        del exc_type, traceback
        try:
            self.close(exit_code=0 if exc is None else 1)
        except BaseException:
            if exc is None:
                raise
            LOGGER.exception("Tracking cleanup failed while handling another exception")
        return False


def create_tracking(
    config: AppConfig,
    directory: str | Path,
    spec: TrackingRunSpec,
) -> ScalarTracking:
    """Create enabled scalar backends without importing optional SDKs unnecessarily."""

    output_directory = Path(directory)
    tensorboard_writer: Any | None = None
    if config.logging.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_writer = SummaryWriter(log_dir=str(output_directory / "tensorboard"))

    wandb_run: Any | None = None
    if config.logging.wandb:
        try:
            import wandb
        except ImportError as exc:
            if tensorboard_writer is not None:
                tensorboard_writer.close()
            raise TrackingError(
                "W&B logging is enabled but the optional SDK is not installed; "
                "install it with `python -m pip install -e '.[wandb]'`"
            ) from exc

        online = config.logging.wandb_mode == "online"
        run_identifier = f"{spec.job_type}-{spec.fingerprint}" if online else None
        run_name = f"{config.dataset.name}-{spec.job_type}-{spec.fingerprint[:8]}"
        run_config = config.model_dump(mode="json")
        run_config["_tracking"] = {
            "fingerprint": spec.fingerprint,
            "job_type": spec.job_type,
            "group": spec.group,
        }
        init_arguments: dict[str, Any] = {
            "project": config.logging.wandb_project,
            "entity": config.logging.wandb_entity,
            "mode": config.logging.wandb_mode,
            "force": online,
            "name": run_name,
            "job_type": spec.job_type,
            "group": spec.group,
            "tags": config.logging.wandb_tags,
            "config": run_config,
            "dir": str(output_directory),
        }
        if run_identifier is not None:
            init_arguments.update(id=run_identifier, resume="allow")
        try:
            wandb_run = wandb.init(**init_arguments)
            if wandb_run is None:
                raise RuntimeError("wandb.init() returned no run")
        except Exception as exc:
            if tensorboard_writer is not None:
                tensorboard_writer.close()
            mode_hint = (
                "run `wandb login` or set WANDB_API_KEY"
                if online
                else "verify that the output directory is writable"
            )
            raise TrackingError(f"W&B initialization failed; {mode_hint}: {exc}") from exc

    try:
        return ScalarTracking(tensorboard_writer, wandb_run)
    except BaseException:
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                LOGGER.exception("W&B cleanup failed after tracking initialization")
        raise


__all__ = [
    "ScalarTracking",
    "TrackingRunSpec",
    "create_tracking",
    "numeric_metrics",
]
