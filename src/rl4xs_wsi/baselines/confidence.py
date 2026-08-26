"""Belief-confidence stopping baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rl4xs_wsi.baselines.base import (
    DIAGNOSE_BENIGN,
    DIAGNOSE_CANCEROUS,
    DIAGNOSIS_ACTIONS,
    BaselinePolicy,
    action_mask,
    belief,
    diagnosis_for_belief,
    valid_navigation,
)
from rl4xs_wsi.baselines.random import RandomPolicy


class ConfidenceThresholdPolicy(BaselinePolicy):
    """Stop when calibrated belief exits ``[1-threshold, threshold]``."""

    def __init__(
        self,
        threshold: float = 0.9,
        *,
        navigation_policy: BaselinePolicy | None = None,
    ) -> None:
        if not 0.5 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0.5,1]")
        self.threshold = float(threshold)
        self.navigation_policy = navigation_policy or RandomPolicy()

    def reset(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        self.navigation_policy.reset(observation, info)

    def act(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
        mask = action_mask(info)
        probability = belief(observation)
        navigation = valid_navigation(mask)
        if mask[DIAGNOSE_CANCEROUS] and probability >= self.threshold:
            return DIAGNOSE_CANCEROUS
        if mask[DIAGNOSE_BENIGN] and probability <= 1.0 - self.threshold:
            return DIAGNOSE_BENIGN
        if not navigation:
            desired = diagnosis_for_belief(probability)
            if mask[desired]:
                return desired
            return next(action for action in DIAGNOSIS_ACTIONS if mask[action])
        action = int(self.navigation_policy.act(observation, info))
        if action in DIAGNOSIS_ACTIONS:
            action = navigation[0]
        if not mask[action]:
            raise RuntimeError(f"wrapped policy selected masked action {action}")
        return action


__all__ = ["ConfidenceThresholdPolicy"]
