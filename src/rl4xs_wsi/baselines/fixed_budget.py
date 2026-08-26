"""Fixed-navigation-budget diagnosis baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rl4xs_wsi.baselines.base import (
    DIAGNOSIS_ACTIONS,
    BaselinePolicy,
    action_mask,
    belief,
    diagnosis_for_belief,
    valid_navigation,
)
from rl4xs_wsi.baselines.random import RandomPolicy


class FixedBudgetPolicy(BaselinePolicy):
    """Delegate navigation and diagnose after an exact action budget."""

    def __init__(
        self,
        budget: int,
        *,
        navigation_policy: BaselinePolicy | None = None,
    ) -> None:
        if isinstance(budget, bool) or budget < 0:
            raise ValueError("budget must be a non-negative integer")
        self.budget = int(budget)
        self.navigation_policy = navigation_policy or RandomPolicy()
        self._steps = 0

    def reset(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        self._steps = 0
        self.navigation_policy.reset(observation, info)

    def act(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
        mask = action_mask(info)
        navigation = valid_navigation(mask)
        diagnosis = diagnosis_for_belief(belief(observation))
        if (self._steps >= self.budget or not navigation) and mask[diagnosis]:
            action = diagnosis
        else:
            action = int(self.navigation_policy.act(observation, info))
            if action in DIAGNOSIS_ACTIONS and self._steps < self.budget and navigation:
                action = navigation[0]
        if not mask[action]:
            raise RuntimeError(f"wrapped policy selected masked action {action}")
        self._steps += 1
        return action


__all__ = ["FixedBudgetPolicy"]
