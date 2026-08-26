"""Read-only environment latency and resource benchmark utility."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rl4xs_wsi.baselines.base import action_mask


@dataclass(frozen=True)
class EnvironmentBenchmark:
    backend: str
    steps: int
    episodes: int
    elapsed_seconds: float
    steps_per_second: float
    median_step_latency_ms: float
    p95_step_latency_ms: float
    mean_reset_latency_ms: float
    p95_reset_latency_ms: float
    process_cpu_percent: float | None
    rss_before_bytes: int | None
    rss_peak_bytes: int | None
    rss_after_bytes: int | None
    gpu_peak_allocated_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _process() -> Any | None:
    try:
        import psutil

        return psutil.Process(os.getpid())
    except ImportError:
        return None


def _rss(process: Any | None) -> int | None:
    return int(process.memory_info().rss) if process is not None else None


def _cpu_seconds(process: Any | None) -> float | None:
    if process is None:
        return None
    times = process.cpu_times()
    return float(times.user + times.system)


def _reset(env: Any, seed: int) -> tuple[Any, Mapping[str, Any], float]:
    started = time.perf_counter()
    observation, info = env.reset(seed=seed)
    elapsed = time.perf_counter() - started
    return observation, info, elapsed


def benchmark_environment(
    env_factory: Callable[[], Any],
    *,
    backend: str,
    steps: int = 1_000,
    seed: int = 0,
) -> EnvironmentBenchmark:
    """Measure masked random interaction without updating any model."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if not backend:
        raise ValueError("backend must be a non-empty label")
    process = _process()
    rss_before = _rss(process)
    rss_peak = rss_before
    cpu_before = _cpu_seconds(process)
    gpu_available = False
    torch_module: Any | None = None
    try:
        import torch as imported_torch

        torch_module = imported_torch
        gpu_available = imported_torch.cuda.is_available()
        if gpu_available:
            imported_torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    env = env_factory()
    latencies: list[float] = []
    reset_latencies: list[float] = []
    episodes = 0
    try:
        observation, info, latency = _reset(env, seed)
        del observation
        reset_latencies.append(latency)
        benchmark_started = time.perf_counter()
        for step_index in range(steps):
            mask = action_mask(info)
            valid = np.flatnonzero(mask)
            action = int(rng.choice(valid))
            started = time.perf_counter()
            observation, _, terminated, truncated, info = env.step(action)
            del observation
            latencies.append(time.perf_counter() - started)
            current_rss = _rss(process)
            if current_rss is not None:
                rss_peak = current_rss if rss_peak is None else max(rss_peak, current_rss)
            if terminated or truncated:
                episodes += 1
                _, info, latency = _reset(env, seed + step_index + 1)
                reset_latencies.append(latency)
        elapsed = time.perf_counter() - benchmark_started
    finally:
        env.close()
    cpu_after = _cpu_seconds(process)
    rss_after = _rss(process)
    cpu_percent = (
        100.0 * (cpu_after - cpu_before) / elapsed
        if cpu_before is not None and cpu_after is not None and elapsed > 0.0
        else None
    )
    gpu_peak = (
        int(torch_module.cuda.max_memory_allocated())
        if gpu_available and torch_module is not None
        else None
    )
    latencies_ms = np.asarray(latencies, dtype=np.float64) * 1_000.0
    resets_ms = np.asarray(reset_latencies, dtype=np.float64) * 1_000.0
    return EnvironmentBenchmark(
        backend=backend,
        steps=steps,
        episodes=episodes,
        elapsed_seconds=float(elapsed),
        steps_per_second=float(steps / elapsed),
        median_step_latency_ms=float(np.median(latencies_ms)),
        p95_step_latency_ms=float(np.percentile(latencies_ms, 95)),
        mean_reset_latency_ms=float(np.mean(resets_ms)),
        p95_reset_latency_ms=float(np.percentile(resets_ms, 95)),
        process_cpu_percent=cpu_percent,
        rss_before_bytes=rss_before,
        rss_peak_bytes=rss_peak,
        rss_after_bytes=rss_after,
        gpu_peak_allocated_bytes=gpu_peak,
    )


def benchmark_backends(
    factories: Mapping[str, Callable[[], Any]],
    *,
    steps: int = 1_000,
    seed: int = 0,
) -> dict[str, EnvironmentBenchmark]:
    if not factories:
        raise ValueError("at least one backend factory is required")
    return {
        name: benchmark_environment(factory, backend=name, steps=steps, seed=seed)
        for name, factory in factories.items()
    }


__all__ = ["EnvironmentBenchmark", "benchmark_backends", "benchmark_environment"]
