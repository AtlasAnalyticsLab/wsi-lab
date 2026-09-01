"""Deterministic accepted-transition scheduling for the WSI-Lab action space."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks.histogym_comparison.common.protocol import (
    EXPECTED_QUOTAS,
    PAN_PREFERENCE,
    abstract_priorities,
    validate_abstract_sequence,
)
from wsi_lab.envs.actions import Action

_PAN_BY_NAME = {
    "up": Action.PAN_UP,
    "down": Action.PAN_DOWN,
    "left": Action.PAN_LEFT,
    "right": Action.PAN_RIGHT,
}
_ZOOM_IN_ORDER = (
    Action.ZOOM_IN_TL,
    Action.ZOOM_IN_TR,
    Action.ZOOM_IN_BL,
    Action.ZOOM_IN_BR,
)


@dataclass(frozen=True, slots=True)
class ActionDecision:
    priority_index: int
    postponement_count: int
    priority_head_class: str
    requested_class: str
    concrete_action: Action
    pending_index: int
    postponed_classes: tuple[str, ...]
    postponed_items: tuple[dict[str, int | str], ...]


@dataclass(slots=True)
class _PriorityItem:
    priority_index: int
    requested_class: str
    postponement_count: int = 0
    unexpected_no_op_count: int = 0


class AcceptedActionScheduler:
    """Consume a shared priority multiset, postponing unavailable classes.

    A priority item is removed only after the environment confirms a state-changing
    transition. Unexpected no-ops blacklist that concrete action at the current
    state and leave the abstract quota item pending for a later retry.
    """

    def __init__(
        self,
        slide_ordinal: int,
        *,
        initial_priorities: Sequence[str] | None = None,
    ) -> None:
        self.slide_ordinal = int(slide_ordinal)
        if initial_priorities is None:
            priorities = abstract_priorities(self.slide_ordinal)
        else:
            validate_abstract_sequence(initial_priorities)
            priorities = tuple(str(value) for value in initial_priorities)
        self.initial_priorities = priorities
        self.pending = deque(
            _PriorityItem(priority_index=index, requested_class=action_class)
            for index, action_class in enumerate(self.initial_priorities)
        )
        self.accepted_counts: Counter[str] = Counter()
        self._blocked: dict[tuple[int, str], set[int]] = defaultdict(set)
        self._inflight: ActionDecision | None = None
        self._inflight_item: _PriorityItem | None = None
        self.no_op_attempts = 0

    @property
    def accepted(self) -> int:
        return sum(self.accepted_counts.values())

    @property
    def complete(self) -> bool:
        return not self.pending and self._inflight is None

    @staticmethod
    def _rotated(values: tuple[Action, ...], amount: int) -> tuple[Action, ...]:
        offset = amount % len(values)
        return values[offset:] + values[:offset]

    def _preference(self, action_class: str) -> tuple[Action, ...]:
        if action_class == "pan":
            base = tuple(_PAN_BY_NAME[name] for name in PAN_PREFERENCE)
            return self._rotated(base, self.accepted_counts["pan"])
        if action_class == "zoom_in":
            return self._rotated(_ZOOM_IN_ORDER, self.accepted_counts["zoom_in"])
        if action_class == "zoom_out":
            return (Action.ZOOM_OUT,)
        raise ValueError(f"unknown abstract action class: {action_class!r}")

    def _candidate(
        self,
        action_class: str,
        action_mask: np.ndarray,
        cell_id: int,
    ) -> Action | None:
        blocked = self._blocked[(int(cell_id), action_class)]
        for action in self._preference(action_class):
            if bool(action_mask[int(action)]) and int(action) not in blocked:
                return action
        return None

    def choose(self, info: dict[str, Any]) -> ActionDecision:
        if self._inflight is not None:
            raise RuntimeError("confirm or reject the previous scheduler decision first")
        if not self.pending:
            raise RuntimeError("all accepted-action quotas are already complete")
        mask = np.asarray(info["action_mask"], dtype=np.bool_)
        if mask.shape != (11,):
            raise ValueError(f"expected action_mask shape (11,), got {mask.shape}")
        cell_id = int(info["cell_id"])
        head = self.pending[0].requested_class
        postponed: list[dict[str, int | str]] = []
        pending_count = len(self.pending)
        for pending_index in range(pending_count):
            item = self.pending.popleft()
            action_class = item.requested_class
            candidate = self._candidate(action_class, mask, cell_id)
            if candidate is not None:
                decision = ActionDecision(
                    priority_index=item.priority_index,
                    postponement_count=item.postponement_count,
                    priority_head_class=head,
                    requested_class=action_class,
                    concrete_action=candidate,
                    pending_index=pending_index,
                    postponed_classes=tuple(
                        str(postponed_item["requested_class"])
                        for postponed_item in postponed
                    ),
                    postponed_items=tuple(postponed),
                )
                self._inflight = decision
                self._inflight_item = item
                return decision
            item.postponement_count += 1
            postponement: dict[str, int | str] = {
                "priority_index": item.priority_index,
                "requested_class": item.requested_class,
                "postponement_count": item.postponement_count,
                "reason": "no_valid_state_changing_concrete_action",
            }
            postponed.append(postponement)
            self.pending.append(item)
        raise RuntimeError(
            "scheduler deadlock: no pending abstract action has an unblocked, "
            "masked concrete transition at cell "
            f"{cell_id}; remaining={Counter(item.requested_class for item in self.pending)}"
        )

    def confirm(self, decision: ActionDecision) -> None:
        if self._inflight != decision:
            raise AssertionError("scheduler decision changed before confirmation")
        self.accepted_counts[decision.requested_class] += 1
        self._inflight = None
        self._inflight_item = None

    def reject_no_op(self, decision: ActionDecision, *, cell_id: int) -> None:
        if self._inflight != decision:
            raise AssertionError("scheduler decision changed before no-op rejection")
        self.no_op_attempts += 1
        self._blocked[(int(cell_id), decision.requested_class)].add(
            int(decision.concrete_action)
        )
        # Retry the same abstract item first. If every concrete action of that
        # class is now unavailable, choose() rotates it to the back exactly as
        # it does any other unavailable priority item.
        if self._inflight_item is None:
            raise AssertionError("scheduler lost the in-flight priority item")
        self._inflight_item.unexpected_no_op_count += 1
        self.pending.appendleft(self._inflight_item)
        self._inflight = None
        self._inflight_item = None

    def verify_complete(self) -> None:
        if self.pending or self._inflight is not None or self._inflight_item is not None:
            raise AssertionError(f"scheduler has {len(self.pending)} pending priorities")
        observed = {name: self.accepted_counts[name] for name in EXPECTED_QUOTAS}
        if observed != EXPECTED_QUOTAS:
            raise AssertionError(f"accepted quotas differ: {observed} != {EXPECTED_QUOTAS}")


def abstract_class(action: Action) -> str:
    if int(action) <= int(Action.PAN_RIGHT):
        return "pan"
    if int(action) <= int(Action.ZOOM_IN_BR):
        return "zoom_in"
    if action is Action.ZOOM_OUT:
        return "zoom_out"
    return "diagnosis"


__all__ = ["AcceptedActionScheduler", "ActionDecision", "abstract_class"]
