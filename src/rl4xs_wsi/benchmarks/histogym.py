"""Optional HistoGym availability and benchmark hooks."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

from rl4xs_wsi.benchmarks.environment import EnvironmentBenchmark, benchmark_environment


def histogym_available() -> bool:
    return importlib.util.find_spec("histogym") is not None


def require_histogym() -> None:
    if not histogym_available():
        raise ImportError(
            "HistoGym is not installed. Install the optional HistoGym dependency "
            "before requesting this comparison."
        )


def benchmark_histogym(
    env_factory: Callable[[], Any], *, steps: int = 1_000, seed: int = 0
) -> EnvironmentBenchmark:
    """Benchmark a caller-supplied HistoGym-compatible Gymnasium factory."""

    require_histogym()
    return benchmark_environment(env_factory, backend="histogym", steps=steps, seed=seed)


__all__ = ["benchmark_histogym", "histogym_available", "require_histogym"]
