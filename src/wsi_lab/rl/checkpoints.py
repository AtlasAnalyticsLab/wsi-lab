"""Atomic recurrent-PPO checkpoint persistence."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    global_step: int
    update: int
    config: Mapping[str, Any]
    dependencies: Mapping[str, str]
    extra: Mapping[str, Any]


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _plain_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if is_dataclass(config):
        return asdict(config)  # type: ignore[arg-type]
    if isinstance(config, Mapping):
        return dict(config)
    return {name: value for name, value in vars(config).items() if not name.startswith("_")}


def save_ppo_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    *,
    global_step: int,
    update: int,
    scheduler: Any | None = None,
    dependencies: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a complete checkpoint in the target directory."""

    if global_step < 0 or update < 0:
        raise ValueError("global_step and update must be non-negative")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "complete": True,
        "global_step": int(global_step),
        "update": int(update),
        "model": model.state_dict(),
        "model_metadata": {
            "class": type(model).__name__,
            "hidden_size": getattr(model, "hidden_size", None),
            "recurrent_layers": getattr(model, "recurrent_layers", None),
            "num_actions": getattr(model, "num_actions", None),
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng_state(),
        "config": _plain_config(config),
        "dependencies": dict(dependencies or {}),
        "extra": dict(extra or {}),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _read_checkpoint_payload(
    source: Path, *, map_location: torch.device | str
) -> Mapping[str, Any]:
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except Exception as exc:
        # torch.load uses several exception families for damaged/truncated files
        # (including EOFError and pickle.UnpicklingError).  Normalize all ordinary
        # load failures so candidate probing can safely try an older checkpoint.
        raise ValueError(f"Could not read PPO checkpoint {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"PPO checkpoint {source} is not a mapping")
    return payload


def _checkpoint_metadata(payload: Mapping[str, Any], source: Path) -> LoadedCheckpoint:
    """Validate the non-model contract and return lightweight checkpoint metadata."""

    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported PPO checkpoint schema in {source}")
    if payload.get("complete") is not True:
        raise ValueError(f"PPO checkpoint is incomplete: {source}")
    required = {
        "model",
        "model_metadata",
        "optimizer",
        "global_step",
        "update",
        "rng",
        "config",
        "dependencies",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"PPO checkpoint {source} is missing fields: {sorted(missing)}")

    mapping_fields = (
        "model",
        "model_metadata",
        "optimizer",
        "rng",
        "config",
        "dependencies",
    )
    invalid_mappings = [name for name in mapping_fields if not isinstance(payload[name], Mapping)]
    if invalid_mappings:
        raise ValueError(f"PPO checkpoint {source} has invalid mapping fields: {invalid_mappings}")
    extra = payload.get("extra")
    if extra is not None and not isinstance(extra, Mapping):
        raise ValueError(f"PPO checkpoint {source} has invalid extra metadata")
    scheduler = payload.get("scheduler")
    if scheduler is not None and not isinstance(scheduler, Mapping):
        raise ValueError(f"PPO checkpoint {source} has invalid scheduler state")

    counters: dict[str, int] = {}
    for name in ("global_step", "update"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(
                f"PPO checkpoint {source} has invalid non-negative integer {name}={value!r}"
            )
        counters[name] = int(value)

    rng = payload["rng"]
    missing_rng = {"python", "numpy", "torch_cpu", "torch_cuda"} - set(rng)
    if missing_rng:
        raise ValueError(f"PPO checkpoint {source} has incomplete RNG state: {sorted(missing_rng)}")
    metadata = payload["model_metadata"]
    if not isinstance(metadata.get("class"), str):
        raise ValueError(f"PPO checkpoint {source} has invalid model class metadata")

    stored_config = payload["config"]
    stored_dependencies = payload["dependencies"]
    return LoadedCheckpoint(
        path=source,
        global_step=counters["global_step"],
        update=counters["update"],
        config=dict(stored_config),
        dependencies={str(key): str(value) for key, value in stored_dependencies.items()},
        extra=dict(extra or {}),
    )


def inspect_ppo_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Validate a checkpoint payload without materializing its tensor storage.

    This is intended for non-mutating pipeline planning.  Model, optimizer, and
    dependency compatibility with a particular run remain loader responsibilities.
    """

    source = Path(path).expanduser().resolve()
    payload = _read_checkpoint_payload(source, map_location="meta")
    return _checkpoint_metadata(payload, source)


def load_ppo_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: Any | None = None,
    map_location: torch.device | str = "cpu",
    restore_rng: bool = True,
    expected_config: Mapping[str, Any] | None = None,
    compatibility_fields: tuple[str, ...] = (),
    expected_dependencies: Mapping[str, str] | None = None,
) -> LoadedCheckpoint:
    source = Path(path).expanduser().resolve()
    payload = _read_checkpoint_payload(source, map_location=map_location)
    loaded = _checkpoint_metadata(payload, source)
    stored_config = loaded.config
    stored_dependencies = loaded.dependencies
    if expected_config is not None:
        config_mismatches = {
            name: (stored_config.get(name), expected_config.get(name))
            for name in compatibility_fields
            if stored_config.get(name) != expected_config.get(name)
        }
        if config_mismatches:
            raise ValueError(f"PPO checkpoint configuration mismatch: {config_mismatches}")
    if expected_dependencies is not None:
        dependency_mismatches = {
            name: (stored_dependencies.get(name), expected)
            for name, expected in expected_dependencies.items()
            if stored_dependencies.get(name) != expected
        }
        unexpected = sorted(set(stored_dependencies) - set(expected_dependencies))
        if dependency_mismatches or unexpected:
            raise ValueError(
                "PPO checkpoint dependency mismatch: "
                f"values={dependency_mismatches}, unexpected={unexpected}"
            )
    metadata = payload["model_metadata"]
    expected_metadata = {
        "hidden_size": getattr(model, "hidden_size", None),
        "recurrent_layers": getattr(model, "recurrent_layers", None),
        "num_actions": getattr(model, "num_actions", None),
    }
    mismatches = {
        name: (metadata.get(name), expected)
        for name, expected in expected_metadata.items()
        if expected is not None and metadata.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"PPO checkpoint model metadata mismatch: {mismatches}")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        model_device = next(model.parameters()).device
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(model_device)
    scheduler_state = payload.get("scheduler")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    if restore_rng:
        restore_rng_state(payload["rng"])
    return loaded


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    candidates = checkpoint_candidates(directory)
    return candidates[0] if candidates else None


def checkpoint_candidates(directory: str | Path) -> list[Path]:
    """Return checkpoint files newest first for compatible-resume probing."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return []
    candidates = [path for path in root.glob("checkpoint_*.pt") if path.is_file()]
    return sorted(
        candidates,
        # Zero-padded trainer filenames sort by global step.  mtime breaks ties
        # for any externally created duplicate naming scheme.
        key=lambda path: (path.name, path.stat().st_mtime_ns),
        reverse=True,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "LoadedCheckpoint",
    "capture_rng_state",
    "checkpoint_candidates",
    "find_latest_checkpoint",
    "inspect_ppo_checkpoint",
    "load_ppo_checkpoint",
    "restore_rng_state",
    "save_ppo_checkpoint",
]
