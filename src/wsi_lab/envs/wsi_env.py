"""Gymnasium environment for dense geometric WSI navigation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from wsi_lab.constants import BELIEF_MAGNIFICATIONS, NAVIGATION_MAGNIFICATIONS, NUM_ACTIONS
from wsi_lab.envs.actions import DIAGNOSIS_ACTIONS, NAVIGATION_ACTIONS, Action
from wsi_lab.envs.observation import (
    OverviewTransform,
    increment_visits,
    location_mask,
    make_observation_space,
    rgb_chw_float,
)
from wsi_lab.envs.rewards import diagnosis_reward, navigation_reward


@dataclass(frozen=True, slots=True)
class NavigationCell:
    cell_id: int
    magnification: float
    row: int
    column: int
    bounds: tuple[int, int, int, int]
    parent_id: int = -1
    child_ids: tuple[int, int, int, int] = (-1, -1, -1, -1)
    neighbor_ids: tuple[int, int, int, int] = (-1, -1, -1, -1)
    tissue_fraction: float = 0.0
    embedding_index: int = -1


@dataclass(slots=True)
class LoadedSlide:
    slide_id: str
    label: int
    overview: np.ndarray
    overview_transform: OverviewTransform
    cells: Mapping[int, NavigationCell]
    start_cell_id: int


class EpisodeSource(Protocol):
    def slide_ids(self, split: str) -> Sequence[str]: ...

    def load_slide(self, slide_id: str) -> LoadedSlide: ...

    def read_viewport(
        self, slide_id: str, cell: NavigationCell, output_size: int
    ) -> np.ndarray: ...

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray: ...


class BeliefState(Protocol):
    @property
    def probability(self) -> float: ...

    def reset(self) -> object: ...

    def add(self, feature: np.ndarray, magnification: float) -> object: ...


class ConstantBelief:
    """Safe empty-bag implementation used when a model is not attached."""

    def __init__(self) -> None:
        self._probability = 0.5

    @property
    def probability(self) -> float:
        return self._probability

    def reset(self) -> None:
        self._probability = 0.5

    def add(self, feature: np.ndarray, magnification: float) -> float:
        del feature, magnification
        return self._probability


class WSIEnv(gym.Env[dict[str, np.ndarray], int]):
    """Dense navigation environment; embedding availability affects belief only."""

    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": ["rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        source: EpisodeSource,
        belief: BeliefState | None = None,
        *,
        split: str = "train",
        evaluation: bool = False,
        overview_size: int = 224,
        viewport_size: int = 224,
        max_episode_steps: int = 500,
        min_diagnostic_evidence: int = 4,
        max_visit_count: int = 10,
        reward_type: str = "default",
        step_cost: float = 0.001,
        belief_reward_beta: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.source = source
        self.belief_state = belief or ConstantBelief()
        self.split = split
        self.evaluation = evaluation
        self.overview_size = overview_size
        self.viewport_size = viewport_size
        self.max_episode_steps = max_episode_steps
        self.min_diagnostic_evidence = min_diagnostic_evidence
        self.max_visit_count = max_visit_count
        self.reward_type = str(getattr(reward_type, "value", reward_type))
        self.step_cost = step_cost
        self.belief_reward_beta = belief_reward_beta
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = make_observation_space(overview_size, viewport_size)
        self._slide_ids = tuple(source.slide_ids(split))
        if not self._slide_ids:
            raise ValueError(f"No slides available for split {split!r}")
        self._evaluation_index = 0
        self._slide: LoadedSlide | None = None
        self._cell: NavigationCell | None = None
        self._visits: np.ndarray | None = None
        self._viewport: np.ndarray | None = None
        self._evidence: set[tuple[float, int]] = set()
        self._actions_taken = 0
        self._terminated = False
        self._visited_cell_ids: set[int] = set()
        self._revisit_count = 0
        self._magnification_histogram: Counter[str] = Counter()
        self._trajectory: list[dict[str, Any]] = []
        self._np_random = np.random.default_rng()

    @property
    def remaining_actions(self) -> int:
        return max(0, self.max_episode_steps - self._actions_taken)

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

    def _select_slide_id(self, options: dict[str, Any] | None) -> str:
        if options and "slide_id" in options:
            slide_id = str(options["slide_id"])
            if slide_id not in self._slide_ids:
                raise ValueError(f"Slide {slide_id!r} is not in split {self.split!r}")
            return slide_id
        if self.evaluation:
            slide_id = self._slide_ids[self._evaluation_index % len(self._slide_ids)]
            self._evaluation_index += 1
            return slide_id
        rng = self._np_random
        assert rng is not None
        return str(rng.choice(self._slide_ids))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._slide = self.source.load_slide(self._select_slide_id(options))
        if self._slide.overview.shape[:2] != (self.overview_size, self.overview_size):
            raise ValueError(
                f"Overview has shape {self._slide.overview.shape}; expected "
                f"({self.overview_size}, {self.overview_size}, 3)"
            )
        self._cell = self._slide.cells[self._slide.start_cell_id]
        if self._cell.magnification != NAVIGATION_MAGNIFICATIONS[0]:
            raise ValueError("Start cell must be at 0.625x")
        self._visits = np.zeros((self.overview_size, self.overview_size), dtype=np.uint32)
        self._viewport = self.source.read_viewport(
            self._slide.slide_id, self._cell, self.viewport_size
        )
        self._evidence.clear()
        self._actions_taken = 0
        self._terminated = False
        self.belief_state.reset()
        self._visited_cell_ids = {self._cell.cell_id}
        self._revisit_count = 0
        self._magnification_histogram = Counter({f"{self._cell.magnification:g}": 1})
        self._trajectory = [self._trajectory_point(action=None)]
        increment_visits(self._visits, self._slide.overview_transform, self._cell.bounds)
        info = self._info()
        info["reset_visit"] = True
        return self._observation(), info

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        if self._terminated:
            return mask
        assert self._cell is not None
        if self.remaining_actions <= 1:
            mask[Action.DIAGNOSE_CANCEROUS] = True
            mask[Action.DIAGNOSE_BENIGN] = True
            return mask
        for index, cell_id in enumerate(self._cell.neighbor_ids):
            mask[index] = cell_id >= 0
        for index, cell_id in enumerate(self._cell.child_ids, start=4):
            mask[index] = cell_id >= 0
        mask[Action.ZOOM_OUT] = self._cell.parent_id >= 0
        if self.evidence_count >= self.min_diagnostic_evidence:
            mask[Action.DIAGNOSE_CANCEROUS] = True
            mask[Action.DIAGNOSE_BENIGN] = True
        return mask

    def _resolve_navigation(self, action: Action) -> NavigationCell:
        assert self._slide is not None and self._cell is not None
        if action.value <= Action.PAN_RIGHT:
            target = self._cell.neighbor_ids[action.value]
        elif action.value <= Action.ZOOM_IN_BR:
            target = self._cell.child_ids[action.value - Action.ZOOM_IN_TL]
        elif action is Action.ZOOM_OUT:
            target = self._cell.parent_id
        else:
            raise ValueError(f"Not a navigation action: {action}")
        return self._slide.cells[target]

    def _maybe_add_evidence(self) -> bool:
        assert self._slide is not None and self._cell is not None
        key = (self._cell.magnification, self._cell.cell_id)
        if (
            self._cell.magnification not in BELIEF_MAGNIFICATIONS
            or self._cell.embedding_index < 0
            or key in self._evidence
        ):
            return False
        feature = self.source.read_embedding(self._slide.slide_id, self._cell)
        self.belief_state.add(feature, self._cell.magnification)
        self._evidence.add(key)
        return True

    def step(self, action: int):
        if self._slide is None or self._cell is None or self._visits is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("step() called after termination; reset the environment")
        try:
            selected = Action(int(action))
        except ValueError as exc:
            raise ValueError(f"Action must be in [0, {NUM_ACTIONS - 1}], got {action}") from exc
        historical_mask = self.action_mask()
        if not historical_mask[selected]:
            info = self._info()
            info.update({"invalid_action": True, "historical_action_mask": historical_mask.copy()})
            return self._observation(), -self.step_cost, False, False, info

        belief_before = float(self.belief_state.probability)
        self._actions_taken += 1
        if selected in DIAGNOSIS_ACTIONS:
            prediction = int(selected is Action.DIAGNOSE_CANCEROUS)
            self._terminated = True
            self._trajectory.append(self._trajectory_point(action=int(selected)))
            reward = diagnosis_reward(prediction, self._slide.label, self.step_cost)
            info = self._info()
            info.update(
                {
                    "diagnosis": prediction,
                    "correct": prediction == self._slide.label,
                    "invalid_action": False,
                    "historical_action_mask": historical_mask.copy(),
                }
            )
            return self._observation(), reward, True, False, info

        if selected not in NAVIGATION_ACTIONS:
            raise AssertionError(f"Unhandled action: {selected}")
        self._cell = self._resolve_navigation(selected)
        revisited = self._cell.cell_id in self._visited_cell_ids
        if revisited:
            self._revisit_count += 1
        self._visited_cell_ids.add(self._cell.cell_id)
        self._magnification_histogram[f"{self._cell.magnification:g}"] += 1
        self._viewport = self.source.read_viewport(
            self._slide.slide_id, self._cell, self.viewport_size
        )
        increment_visits(self._visits, self._slide.overview_transform, self._cell.bounds)
        evidence_added = self._maybe_add_evidence()
        self._trajectory.append(self._trajectory_point(action=int(selected)))
        belief_after = float(self.belief_state.probability)
        reward = navigation_reward(
            label=self._slide.label,
            belief_before=belief_before,
            belief_after=belief_after,
            reward_type=self.reward_type,
            step_cost=self.step_cost,
            beta=self.belief_reward_beta,
        )
        info = self._info()
        info.update(
            {
                "evidence_added": evidence_added,
                "invalid_action": False,
                "historical_action_mask": historical_mask.copy(),
            }
        )
        return self._observation(), reward, False, False, info

    def _observation(self) -> dict[str, np.ndarray]:
        assert self._slide is not None and self._cell is not None
        assert self._visits is not None and self._viewport is not None
        overview = rgb_chw_float(self._slide.overview)
        location = location_mask(self._slide.overview_transform, self._cell.bounds)
        visits = np.minimum(self._visits, self.max_visit_count).astype(np.float32)
        visits /= float(self.max_visit_count)
        magnification_index = NAVIGATION_MAGNIFICATIONS.index(self._cell.magnification)
        scalars = np.asarray(
            [
                magnification_index / (len(NAVIGATION_MAGNIFICATIONS) - 1),
                self.remaining_actions / self.max_episode_steps,
                float(self.belief_state.probability),
            ],
            dtype=np.float32,
        )
        return {
            "global": np.concatenate((overview, location[None], visits[None]), axis=0).astype(
                np.float32
            ),
            "viewport": rgb_chw_float(self._viewport),
            "scalars": np.clip(scalars, 0.0, 1.0),
        }

    def _info(self) -> dict[str, Any]:
        assert self._slide is not None and self._cell is not None
        observed_views = sum(self._magnification_histogram.values())
        return {
            "slide_id": self._slide.slide_id,
            "label": self._slide.label,
            "cell_id": self._cell.cell_id,
            "magnification": self._cell.magnification,
            "row": self._cell.row,
            "column": self._cell.column,
            "belief": float(self.belief_state.probability),
            "evidence_count": self.evidence_count,
            "unique_evidence_count": self.evidence_count,
            "actions_taken": self._actions_taken,
            "remaining_actions": self.remaining_actions,
            "action_mask": self.action_mask(),
            "magnification_histogram": dict(self._magnification_histogram),
            "revisit_rate": self._revisit_count / observed_views if observed_views else 0.0,
            "inspected_fraction": len(self._visited_cell_ids) / len(self._slide.cells),
        }

    def _trajectory_point(self, action: int | None) -> dict[str, Any]:
        assert self._cell is not None
        return {
            "step": self._actions_taken,
            "action": action,
            "cell_id": self._cell.cell_id,
            "magnification": self._cell.magnification,
            "row": self._cell.row,
            "column": self._cell.column,
            "bounds": list(self._cell.bounds),
            "belief": float(self.belief_state.probability),
            "evidence_count": self.evidence_count,
        }

    @property
    def trajectory(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._trajectory)

    def render(self) -> np.ndarray:
        from wsi_lab.envs.rendering import render_state

        if self._slide is None or self._cell is None or self._visits is None:
            raise RuntimeError("reset() must be called before render()")
        return render_state(
            self._slide.overview,
            self._slide.overview_transform,
            self._cell.bounds,
            self._visits,
            self.max_visit_count,
        )

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if callable(close):
            close()


def make_env(source: EpisodeSource | None = None, belief: BeliefState | None = None, **kwargs: Any):
    if source is None:
        from wsi_lab.testing.synthetic import SyntheticEpisodeSource

        overview_size = int(kwargs.get("overview_size", 224))
        source = SyntheticEpisodeSource(overview_size=overview_size)
    return WSIEnv(source=source, belief=belief, **kwargs)
