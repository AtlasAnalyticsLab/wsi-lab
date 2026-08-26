"""Deterministic serpentine raster-scanning baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wsi_lab.baselines.base import (
    DIAGNOSIS_ACTIONS,
    NAVIGATION_ACTIONS,
    ZOOM_IN_ACTIONS,
    BaselinePolicy,
    action_mask,
    belief,
    diagnosis_for_belief,
)


class RasterPolicy(BaselinePolicy):
    """Zoom to a target scale, then scan rows in alternating directions."""

    def __init__(self, *, target_magnification: float = 1.0) -> None:
        if not 0.0 <= target_magnification <= 1.0:
            raise ValueError("target_magnification must lie in [0,1]")
        self.target_magnification = float(target_magnification)
        self._horizontal_action = 3  # PAN_RIGHT

    def reset(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        self._horizontal_action = 3

    def act(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
        mask = action_mask(info)
        scalar_values = observation["scalars"]
        current_magnification = float(scalar_values[0])
        if current_magnification + 1e-6 < self.target_magnification:
            for action in ZOOM_IN_ACTIONS:
                if mask[action]:
                    return action

        if mask[self._horizontal_action]:
            return self._horizontal_action
        if mask[1]:  # PAN_DOWN: advance one row, then reverse direction.
            self._horizontal_action = 2 if self._horizontal_action == 3 else 3
            return 1

        desired = diagnosis_for_belief(belief(observation))
        if mask[desired]:
            return desired
        for action in NAVIGATION_ACTIONS:
            if mask[action]:
                return action
        for action in DIAGNOSIS_ACTIONS:
            if mask[action]:
                return action
        raise RuntimeError("environment exposed no valid raster action")


__all__ = ["RasterPolicy"]
