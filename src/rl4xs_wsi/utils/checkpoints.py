"""Checkpoint composition and compatibility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl4xs_wsi.errors import ArtifactCompatibilityError
from rl4xs_wsi.utils.atomic import atomic_torch_save
from rl4xs_wsi.utils.seeds import capture_rng_state


def save_training_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    global_step: int,
    epoch: int,
    config: dict[str, Any],
    dependencies: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> None:
    atomic_torch_save(
        path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
            "epoch": epoch,
            "rng_state": capture_rng_state(),
            "config": config,
            "dependencies": dependencies,
            "extra": extra or {},
        },
    )


def require_dependencies(checkpoint: dict[str, Any], expected: dict[str, str]) -> None:
    actual = checkpoint.get("dependencies", {})
    mismatches = {
        key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value
    }
    if mismatches:
        raise ArtifactCompatibilityError(f"Checkpoint dependency mismatch: {mismatches}")
