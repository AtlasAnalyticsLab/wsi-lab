"""Low-overhead process-tree and optional NVIDIA device resource sampling."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil


@dataclass(frozen=True)
class _Counters:
    cpu_seconds: float
    read_bytes: int
    write_bytes: int
    read_chars: int
    write_chars: int
    read_count: int
    write_count: int


def _zero_counters() -> _Counters:
    return _Counters(0.0, 0, 0, 0, 0, 0, 0)


def _counter_for(process: psutil.Process) -> _Counters:
    cpu = process.cpu_times()
    try:
        io = process.io_counters()
    except (psutil.AccessDenied, AttributeError):
        return _Counters(cpu.user + cpu.system, 0, 0, 0, 0, 0, 0)
    return _Counters(
        cpu.user + cpu.system,
        io.read_bytes,
        io.write_bytes,
        io.read_chars,
        io.write_chars,
        io.read_count,
        io.write_count,
    )


def _subtract(
    after: _Counters, before: _Counters
) -> Dict[str, Union[float, int]]:
    return {
        "cpu_seconds": max(0.0, after.cpu_seconds - before.cpu_seconds),
        "read_bytes": max(0, after.read_bytes - before.read_bytes),
        "write_bytes": max(0, after.write_bytes - before.write_bytes),
        "read_chars": max(0, after.read_chars - before.read_chars),
        "write_chars": max(0, after.write_chars - before.write_chars),
        "read_syscalls": max(0, after.read_count - before.read_count),
        "write_syscalls": max(0, after.write_count - before.write_count),
    }


class ResourceMonitor:
    """Sample the current process tree and one whole GPU during a timed phase."""

    def __init__(
        self,
        interval_seconds: float = 0.10,
        gpu_index: Optional[int] = None,
        gpu_interval_seconds: float = 1.0,
    ):
        self.interval_seconds = interval_seconds
        self.gpu_index = gpu_index
        self.gpu_interval_seconds = gpu_interval_seconds
        self.root = psutil.Process(os.getpid())
        self._process_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_ns = 0
        self._ended_ns = 0
        self._initial: Dict[Tuple[int, float], _Counters] = {}
        self._maximum: Dict[Tuple[int, float], _Counters] = {}
        self._samples = 0
        self._peak_rss = 0
        self._baseline_rss = 0
        self._peak_gpu_memory_mib: Optional[float] = None
        self._baseline_gpu_memory_mib: Optional[float] = None
        self._peak_gpu_utilization_percent: Optional[float] = None
        self._gpu_utilization_sum = 0.0
        self._gpu_samples = 0
        self._peak_gpu_power_watts: Optional[float] = None
        self._last_gpu_sample = 0.0

    def _processes(self) -> List[psutil.Process]:
        try:
            return [self.root, *self.root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    @staticmethod
    def _key(process: psutil.Process) -> Tuple[int, float]:
        return process.pid, process.create_time()

    def _sample_processes(self, initial: bool = False) -> None:
        with self._process_lock:
            rss = 0
            for process in self._processes():
                try:
                    key = self._key(process)
                    counters = _counter_for(process)
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                    continue
                if initial and key not in self._initial:
                    self._initial[key] = counters
                elif key not in self._initial:
                    self._initial[key] = _zero_counters()
                previous = self._maximum.get(key, _zero_counters())
                self._maximum[key] = _Counters(
                    max(previous.cpu_seconds, counters.cpu_seconds),
                    max(previous.read_bytes, counters.read_bytes),
                    max(previous.write_bytes, counters.write_bytes),
                    max(previous.read_chars, counters.read_chars),
                    max(previous.write_chars, counters.write_chars),
                    max(previous.read_count, counters.read_count),
                    max(previous.write_count, counters.write_count),
                )
            self._peak_rss = max(self._peak_rss, rss)
            if initial:
                self._baseline_rss = rss
            self._samples += 1

    def prefix_rss_snapshot(self) -> Dict[str, int]:
        """Return the monitor-sampled RSS prefix without stopping the monitor."""

        if self._thread is None or self._stop.is_set():
            raise RuntimeError("ResourceMonitor must be running for a prefix snapshot")
        # Force the episode-boundary RSS into the same peak state used by the
        # asynchronous samples and the final stop() result.
        self._sample_processes()
        with self._process_lock:
            return {
                "baseline_rss_bytes": self._baseline_rss,
                "cumulative_peak_rss_bytes": self._peak_rss,
                "cumulative_incremental_peak_rss_bytes": max(
                    0, self._peak_rss - self._baseline_rss
                ),
                "process_samples": self._samples,
            }

    def _sample_gpu(self, *, force: bool = False) -> None:
        if self.gpu_index is None:
            return
        now = time.monotonic()
        if not force and now - self._last_gpu_sample < self.gpu_interval_seconds:
            return
        self._last_gpu_sample = now
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(1.0, self.interval_seconds * 10),
            )
            utilization, memory, power = (
                float(field.strip()) for field in completed.stdout.strip().split(",")
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        self._peak_gpu_utilization_percent = max(
            self._peak_gpu_utilization_percent or 0.0, utilization
        )
        self._peak_gpu_memory_mib = max(self._peak_gpu_memory_mib or 0.0, memory)
        self._peak_gpu_power_watts = max(self._peak_gpu_power_watts or 0.0, power)
        self._gpu_utilization_sum += utilization
        self._gpu_samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_processes()
            self._sample_gpu()

    def start(self) -> "ResourceMonitor":
        if self._thread is not None:
            raise RuntimeError("ResourceMonitor cannot be started twice")
        self._sample_processes(initial=True)
        self._sample_gpu(force=True)
        self._baseline_gpu_memory_mib = self._peak_gpu_memory_mib
        self._started_ns = time.perf_counter_ns()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("ResourceMonitor was not started")
        self._stop.set()
        self._thread.join()
        self._sample_processes()
        self._ended_ns = time.perf_counter_ns()
        self._sample_gpu(force=True)

        initial_total = _zero_counters()
        maximum_total = _zero_counters()
        for key, maximum in self._maximum.items():
            initial = self._initial.get(key, _zero_counters())
            initial_total = _Counters(
                initial_total.cpu_seconds + initial.cpu_seconds,
                initial_total.read_bytes + initial.read_bytes,
                initial_total.write_bytes + initial.write_bytes,
                initial_total.read_chars + initial.read_chars,
                initial_total.write_chars + initial.write_chars,
                initial_total.read_count + initial.read_count,
                initial_total.write_count + initial.write_count,
            )
            maximum_total = _Counters(
                maximum_total.cpu_seconds + maximum.cpu_seconds,
                maximum_total.read_bytes + maximum.read_bytes,
                maximum_total.write_bytes + maximum.write_bytes,
                maximum_total.read_chars + maximum.read_chars,
                maximum_total.write_chars + maximum.write_chars,
                maximum_total.read_count + maximum.read_count,
                maximum_total.write_count + maximum.write_count,
            )

        wall_seconds = (self._ended_ns - self._started_ns) / 1e9
        delta = _subtract(maximum_total, initial_total)
        logical_cpus = psutil.cpu_count(logical=True) or 1
        cpu_one_core_percent = (
            float(delta["cpu_seconds"]) / wall_seconds * 100.0 if wall_seconds else 0.0
        )
        return {
            "wall_seconds": wall_seconds,
            "cpu_seconds": delta.pop("cpu_seconds"),
            "average_cpu_percent_one_core_100": cpu_one_core_percent,
            "average_cpu_percent_machine_normalized": cpu_one_core_percent / logical_cpus,
            "logical_cpu_count": logical_cpus,
            "baseline_rss_bytes": self._baseline_rss,
            "peak_rss_bytes": self._peak_rss,
            "peak_incremental_rss_bytes": max(0, self._peak_rss - self._baseline_rss),
            "process_samples": self._samples,
            "io": delta,
            "gpu_index": self.gpu_index,
            "gpu_samples": self._gpu_samples,
            "gpu_baseline_memory_mib": self._baseline_gpu_memory_mib,
            "gpu_peak_memory_mib": self._peak_gpu_memory_mib,
            "gpu_peak_incremental_memory_mib": (
                max(0.0, self._peak_gpu_memory_mib - self._baseline_gpu_memory_mib)
                if self._peak_gpu_memory_mib is not None
                and self._baseline_gpu_memory_mib is not None
                else None
            ),
            "gpu_peak_utilization_percent": self._peak_gpu_utilization_percent,
            "gpu_average_utilization_percent": (
                self._gpu_utilization_sum / self._gpu_samples if self._gpu_samples else None
            ),
            "gpu_peak_power_watts": self._peak_gpu_power_watts,
        }

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._thread is not None and not self._stop.is_set():
            self.stop()
