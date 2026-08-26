"""Common stateful interface and helpers for non-learning policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

PAN_ACTIONS = (0, 1, 2, 3)
ZOOM_IN_ACTIONS = (4, 5, 6, 7)
ZOOM_OUT_ACTION = 8
DIAGNOSE_CANCEROUS = 9
DIAGNOSE_BENIGN = 10
DIAGNOSIS_ACTIONS = (DIAGNOSE_CANCEROUS, DIAGNOSE_BENIGN)
NAVIGATION_ACTIONS = tuple(range(9))
NUM_ACTIONS = 11


def action_mask(info: Mapping[str, Any]) -> np.ndarray:
    if "action_mask" not in info:
        raise KeyError("baseline policies require info['action_mask']")
    mask = np.asarray(info["action_mask"], dtype=np.bool_)
    if mask.shape != (NUM_ACTIONS,):
        raise ValueError(f"action_mask must have shape [{NUM_ACTIONS}]")
    if not mask.any():
        raise ValueError("action_mask contains no valid action")
    return mask


def belief(observation: Mapping[str, Any]) -> float:
    scalars = np.asarray(observation["scalars"], dtype=np.float32)
    if scalars.shape != (5,):
        raise ValueError("single-environment scalar observation must have shape [5]")
    value = float(scalars[2])
    if not 0.0 <= value <= 1.0:
        raise ValueError("belief scalar must lie in [0,1]")
    return value


def diagnosis_for_belief(probability: float) -> int:
    return DIAGNOSE_CANCEROUS if probability >= 0.5 else DIAGNOSE_BENIGN


def valid_navigation(mask: np.ndarray) -> list[int]:
    return [action for action in NAVIGATION_ACTIONS if bool(mask[action])]


class BaselinePolicy(ABC):
    """Stateful per-environment policy with the same contract as RL evaluation."""

    def reset(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        """Reset episode-local state. Stateless policies may keep the default."""
        del observation, info

    @abstractmethod
    def act(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
        """Return one valid integer action from the current metadata mask."""


__all__ = [
    "DIAGNOSE_BENIGN",
    "DIAGNOSE_CANCEROUS",
    "DIAGNOSIS_ACTIONS",
    "NAVIGATION_ACTIONS",
    "NUM_ACTIONS",
    "PAN_ACTIONS",
    "ZOOM_IN_ACTIONS",
    "ZOOM_OUT_ACTION",
    "BaselinePolicy",
    "action_mask",
    "belief",
    "diagnosis_for_belief",
    "valid_navigation",
]
