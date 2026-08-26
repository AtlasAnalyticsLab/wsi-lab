"""Reproducible random-number management."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})


def configure_deterministic_cuda() -> str:
    """Set the cuBLAS workspace contract required by deterministic PyTorch.

    This must happen before the first CUDA matrix multiplication. Respect either
    NVIDIA-supported value when it is already configured, and fail clearly for
    an incompatible user value instead of silently overriding it.
    """

    configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if configured not in _CUBLAS_WORKSPACE_CONFIGS:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8' for "
            f"deterministic CUDA execution, got {configured!r}"
        )
    return configured


def seed_everything(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        configure_deterministic_cuda()
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except ImportError:
        pass


__all__ = [
    "capture_rng_state",
    "configure_deterministic_cuda",
    "restore_rng_state",
    "seed_everything",
]
