"""Masked random-navigation baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rl4xs_wsi.baselines.base import (
    DIAGNOSIS_ACTIONS,
    BaselinePolicy,
    action_mask,
    belief,
    diagnosis_for_belief,
    valid_navigation,
)


class RandomPolicy(BaselinePolicy):
    """Navigate uniformly at random, diagnosing only by configured probability."""

    def __init__(self, *, seed: int = 0, voluntary_diagnosis_probability: float = 0.0) -> None:
        if not 0.0 <= voluntary_diagnosis_probability <= 1.0:
            raise ValueError("voluntary_diagnosis_probability must lie in [0,1]")
        self._rng = np.random.default_rng(seed)
        self.voluntary_diagnosis_probability = float(voluntary_diagnosis_probability)

    def act(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
        mask = action_mask(info)
        navigation = valid_navigation(mask)
        diagnosis_valid = any(bool(mask[action]) for action in DIAGNOSIS_ACTIONS)
        if not navigation:
            action = diagnosis_for_belief(belief(observation))
            if not mask[action]:
                action = DIAGNOSIS_ACTIONS[1 - DIAGNOSIS_ACTIONS.index(action)]
            return int(action)
        if diagnosis_valid and self._rng.random() < self.voluntary_diagnosis_probability:
            return diagnosis_for_belief(belief(observation))
        return int(self._rng.choice(navigation))


__all__ = ["RandomPolicy"]
