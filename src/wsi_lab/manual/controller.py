"""Keyboard-to-environment adapter with no duplicated transition rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from wsi_lab.constants import NUM_ACTIONS
from wsi_lab.envs.actions import Action


@dataclass(frozen=True, slots=True)
class ActionKeyBinding:
    key: str
    action: Action
    label: str


ACTION_KEY_BINDINGS: tuple[ActionKeyBinding, ...] = (
    ActionKeyBinding("up", Action.PAN_UP, "pan up"),
    ActionKeyBinding("down", Action.PAN_DOWN, "pan down"),
    ActionKeyBinding("left", Action.PAN_LEFT, "pan left"),
    ActionKeyBinding("right", Action.PAN_RIGHT, "pan right"),
    ActionKeyBinding("7", Action.ZOOM_IN_TL, "zoom top-left"),
    ActionKeyBinding("9", Action.ZOOM_IN_TR, "zoom top-right"),
    ActionKeyBinding("1", Action.ZOOM_IN_BL, "zoom bottom-left"),
    ActionKeyBinding("3", Action.ZOOM_IN_BR, "zoom bottom-right"),
    ActionKeyBinding("-", Action.ZOOM_OUT, "zoom out"),
    ActionKeyBinding("c", Action.DIAGNOSE_CANCEROUS, "diagnose cancerous"),
    ActionKeyBinding("b", Action.DIAGNOSE_BENIGN, "diagnose benign"),
)

_KEY_TO_ACTION: Mapping[str, Action] = {
    binding.key: binding.action for binding in ACTION_KEY_BINDINGS
}
_KEY_ALIASES: Mapping[str, str] = {"backspace": "-"}


@dataclass(frozen=True, slots=True)
class ManualEvent:
    """Result of one keyboard event handled by :class:`ManualController`."""

    kind: str
    message: str
    action: Action | None = None
    reward: float | None = None
    state_changed: bool = False
    close_requested: bool = False


class ManualController:
    """Own manual-play state while delegating all WSI behavior to Gymnasium.

    The controller never computes navigation validity. It consumes the current
    ``info["action_mask"]`` produced by the environment and calls ``step`` only
    when that exact mask permits the selected action.
    """

    def __init__(
        self,
        environment: gym.Env[dict[str, np.ndarray], int],
        *,
        slide_ids: Sequence[str] = (),
        seed: int | None = None,
    ) -> None:
        self.environment = environment
        self.slide_ids = tuple(str(value) for value in slide_ids)
        if len(set(self.slide_ids)) != len(self.slide_ids):
            raise ValueError("manual-play slide IDs must be unique")
        self.seed = seed
        self.observation: dict[str, np.ndarray] | None = None
        self.info: dict[str, Any] | None = None
        self.cumulative_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.latest_action: Action | None = None
        self.latest_reward: float | None = None
        self.latest_message = "not started"
        self.latest_action_blocked = False
        self._has_reset = False

    @property
    def current_slide_id(self) -> str | None:
        if self.info is None:
            return None
        value = self.info.get("slide_id")
        return None if value is None else str(value)

    @property
    def action_mask(self) -> np.ndarray:
        if self.info is None:
            raise RuntimeError("manual controller has not been reset")
        if "action_mask" not in self.info:
            raise KeyError("environment info does not contain action_mask")
        mask = np.asarray(self.info["action_mask"], dtype=np.bool_)
        if mask.shape != (NUM_ACTIONS,):
            raise ValueError(
                f"environment action mask has shape {mask.shape}; expected {(NUM_ACTIONS,)}"
            )
        return mask.copy()

    def reset(self, slide_id: str | None = None) -> ManualEvent:
        if slide_id is not None and self.slide_ids and slide_id not in self.slide_ids:
            raise ValueError(f"slide {slide_id!r} is not available to manual play")
        options = None if slide_id is None else {"slide_id": slide_id}
        reset_seed = self.seed if not self._has_reset else None
        observation, info = self.environment.reset(seed=reset_seed, options=options)
        self.observation = observation
        self.info = dict(info)
        self.cumulative_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.latest_action = None
        self.latest_reward = None
        self.latest_action_blocked = False
        self._has_reset = True
        self.latest_message = f"reset slide {self.current_slide_id}"
        return ManualEvent("reset", self.latest_message, state_changed=True)

    def reset_current(self) -> ManualEvent:
        slide_id = self.current_slide_id
        if slide_id is None:
            return self.reset()
        return self.reset(slide_id)

    def select_relative_slide(self, offset: int) -> ManualEvent:
        if not self.slide_ids:
            return self.reset()
        current = self.current_slide_id
        try:
            index = self.slide_ids.index(current) if current is not None else -1
        except ValueError:
            index = -1
        return self.reset(self.slide_ids[(index + offset) % len(self.slide_ids)])

    def perform_action(self, action: Action | int) -> ManualEvent:
        if self.observation is None or self.info is None:
            raise RuntimeError("manual controller must be reset before taking an action")
        selected = Action(int(action))
        self.latest_action = selected
        self.latest_reward = None
        if self.terminated or self.truncated:
            self.latest_action_blocked = True
            self.latest_message = "episode is complete; reset before acting"
            return ManualEvent("blocked", self.latest_message, action=selected)
        if not self.action_mask[int(selected)]:
            self.latest_action_blocked = True
            self.latest_message = f"blocked invalid action: {selected.name}"
            return ManualEvent("blocked", self.latest_message, action=selected)

        observation, reward, terminated, truncated, info = self.environment.step(int(selected))
        self.observation = observation
        self.info = dict(info)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.latest_reward = float(reward)
        self.cumulative_reward += float(reward)
        self.latest_action_blocked = False
        self.latest_message = f"{selected.name}: reward {float(reward):+.6f}"
        return ManualEvent(
            "action",
            self.latest_message,
            action=selected,
            reward=float(reward),
            state_changed=True,
        )

    def handle_key(self, key: str | None) -> ManualEvent:
        normalized = "" if key is None else str(key).lower()
        normalized = _KEY_ALIASES.get(normalized, normalized)
        if normalized == "escape":
            return ManualEvent("close", "close requested", close_requested=True)
        if normalized == "r":
            return self.reset_current()
        if normalized in {"n", "]"}:
            return self.select_relative_slide(1)
        if normalized == "[":
            return self.select_relative_slide(-1)
        action = _KEY_TO_ACTION.get(normalized)
        if action is None:
            self.latest_message = f"unmapped key: {key!r}"
            return ManualEvent("unmapped", self.latest_message)
        return self.perform_action(action)

    def close(self) -> None:
        self.environment.close()


__all__ = ["ACTION_KEY_BINDINGS", "ActionKeyBinding", "ManualController", "ManualEvent"]
