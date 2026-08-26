from __future__ import annotations

import numpy as np

from rl4xs_wsi.benchmarks.environment import benchmark_environment


class TinyEnvironment:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed=None):
        del seed
        self.steps = 0
        mask = np.zeros(11, dtype=np.bool_)
        mask[0] = True
        return {}, {"action_mask": mask}

    def step(self, action):
        assert action in (0, 9, 10)
        self.steps += 1
        terminal = self.steps == 2
        mask = np.zeros(11, dtype=np.bool_)
        mask[[9, 10] if terminal else [0]] = True
        return {}, -0.001, terminal, False, {"action_mask": mask}

    def close(self):
        pass


def test_environment_benchmark_collects_latency_and_memory_metrics() -> None:
    result = benchmark_environment(TinyEnvironment, backend="synthetic", steps=4)
    assert result.steps == 4
    assert result.episodes == 2
    assert result.steps_per_second > 0.0
    assert result.median_step_latency_ms >= 0.0
