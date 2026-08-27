"""Gymnasium environment for dense geometric WSI navigation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from wsi_lab.constants import (
    BELIEF_MAGNIFICATIONS,
    LEVEL0_FOOTPRINTS,
    NAVIGATION_MAGNIFICATIONS,
    NUM_ACTIONS,
)
from wsi_lab.envs.actions import DIAGNOSIS_ACTIONS, NAVIGATION_ACTIONS, Action
from wsi_lab.envs.augmentation import GeometricOrientation, HEStainAugmentation
from wsi_lab.envs.observation import (
    OverviewTransform,
    ROIObservationTransform,
    crop_overview_to_roi,
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
    # Defaults to true so callers loading navigation artifacts created before
    # tissue-bounded navigation keep the historical all-cells-navigable behavior.
    tissue_valid: bool = True


@dataclass(slots=True)
class LoadedSlide:
    slide_id: str
    label: int
    overview: np.ndarray
    overview_transform: OverviewTransform
    cells: Mapping[int, NavigationCell]
    start_cell_id: int
    # The snapped tissue ROI is expressed in the same level-0/global coordinate
    # system as ``NavigationCell.bounds``.  ``None`` preserves compatibility
    # with custom/legacy episode sources by using the full slide extent.
    navigation_roi_level0: tuple[int, int, int, int] | None = None


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
        start_position: str = "center",
        geometric_augmentation: bool = False,
        photometric_augmentation: bool = False,
        max_episode_steps: int = 500,
        min_diagnostic_evidence: int = 32,
        max_visit_count: int = 10,
        reward_type: str = "default",
        step_cost: float = 0.001,
        belief_reward_beta: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        normalized_start_position = str(getattr(start_position, "value", start_position))
        if normalized_start_position not in ("center", "random"):
            raise ValueError("start_position must be 'center' or 'random'")
        self.source = source
        self.belief_state = belief or ConstantBelief()
        self.split = split
        self.evaluation = evaluation
        self.overview_size = overview_size
        self.viewport_size = viewport_size
        self.start_position = normalized_start_position
        self.geometric_augmentation = bool(geometric_augmentation)
        self.photometric_augmentation = bool(photometric_augmentation)
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
        self._augmentation_frame_size: tuple[int, int] | None = None
        self._canonical_navigation_roi: tuple[int, int, int, int] | None = None
        self._geometric_orientation = GeometricOrientation()
        self._stain_augmentation = HEStainAugmentation()
        self._geometric_random = np.random.default_rng()
        self._photometric_random = np.random.default_rng()
        self._canonical_observation_transform: ROIObservationTransform | None = None
        self._observation_transform: ROIObservationTransform | None = None
        self._canonical_spatial_overview: np.ndarray | None = None
        self._spatial_overview: np.ndarray | None = None
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

    @property
    def observation_transform(self) -> ROIObservationTransform:
        """Map the current agent-facing level-0 frame into the ROI observation."""

        if self._observation_transform is None:
            raise RuntimeError("reset() must be called before accessing the observation transform")
        return self._observation_transform

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

    def _select_start_cell(self) -> NavigationCell:
        assert self._slide is not None
        if self.start_position == "center":
            return self._slide.cells[self._slide.start_cell_id]
        candidates = sorted(
            cell.cell_id
            for cell in self._slide.cells.values()
            if cell.magnification == NAVIGATION_MAGNIFICATIONS[0] and cell.tissue_valid
        )
        if not candidates:
            raise ValueError("Slide has no tissue-valid cells at 0.625x")
        rng = self._np_random
        assert rng is not None
        return self._slide.cells[int(rng.choice(candidates))]

    def _sample_augmentation(self) -> None:
        """Sample one slide-consistent augmentation for the current episode."""

        assert self._slide is not None
        coarse_footprint = LEVEL0_FOOTPRINTS[NAVIGATION_MAGNIFICATIONS[0]]
        source_width = self._slide.overview_transform.source_width
        source_height = self._slide.overview_transform.source_height
        self._augmentation_frame_size = (
            math.ceil(source_width / coarse_footprint) * coarse_footprint,
            math.ceil(source_height / coarse_footprint) * coarse_footprint,
        )
        self._geometric_orientation = (
            GeometricOrientation.sample(self._geometric_random)
            if self.geometric_augmentation
            else GeometricOrientation()
        )
        self._stain_augmentation = (
            HEStainAugmentation.sample(self._photometric_random)
            if self.photometric_augmentation
            else HEStainAugmentation()
        )

    def _agent_bounds(self, bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if self._augmentation_frame_size is None:
            raise RuntimeError("reset() must be called before transforming coordinates")
        width, height = self._augmentation_frame_size
        return self._geometric_orientation.transform_bounds(bounds, width, height)

    def _agent_row_column(self, cell: NavigationCell) -> tuple[int, int]:
        if not self.geometric_augmentation:
            return cell.row, cell.column
        footprint = LEVEL0_FOOTPRINTS[cell.magnification]
        x0, y0, _, _ = self._agent_bounds(cell.bounds)
        return y0 // footprint, x0 // footprint

    def _read_augmented_viewport(self, cell: NavigationCell) -> np.ndarray:
        assert self._slide is not None
        viewport = self.source.read_viewport(self._slide.slide_id, cell, self.viewport_size)
        if self.geometric_augmentation:
            viewport = self._geometric_orientation.apply_image(viewport)
        if self.photometric_augmentation:
            viewport = self._stain_augmentation.apply_image(viewport)
        return np.ascontiguousarray(viewport)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            geometric_seed, photometric_seed = np.random.SeedSequence(seed).spawn(2)
            self._geometric_random = np.random.default_rng(geometric_seed)
            self._photometric_random = np.random.default_rng(photometric_seed)
        self._slide = self.source.load_slide(self._select_slide_id(options))
        self._sample_augmentation()
        if self._slide.overview.shape[:2] != (self.overview_size, self.overview_size):
            raise ValueError(
                f"Overview has shape {self._slide.overview.shape}; expected "
                f"({self.overview_size}, {self.overview_size}, 3)"
            )
        full_slide_bounds = (
            0,
            0,
            self._slide.overview_transform.source_width,
            self._slide.overview_transform.source_height,
        )
        roi_bounds = self._slide.navigation_roi_level0 or full_slide_bounds
        self._canonical_navigation_roi = roi_bounds
        augmented_roi_bounds = self._agent_bounds(roi_bounds)
        self._observation_transform = ROIObservationTransform.fit(
            augmented_roi_bounds,
            (self.overview_size, self.overview_size),
        )
        self._canonical_observation_transform = ROIObservationTransform.fit(
            roi_bounds,
            (self.overview_size, self.overview_size),
        )
        spatial_overview = crop_overview_to_roi(
            self._slide.overview,
            self._slide.overview_transform,
            self._canonical_observation_transform,
        )
        if self.photometric_augmentation:
            spatial_overview = self._stain_augmentation.apply_image(spatial_overview)
        self._canonical_spatial_overview = np.ascontiguousarray(
            spatial_overview,
            dtype=np.uint8,
        )
        self._spatial_overview = (
            self._geometric_orientation.apply_image(self._canonical_spatial_overview)
            if self.geometric_augmentation
            else self._canonical_spatial_overview.copy()
        )
        self._cell = self._select_start_cell()
        if self._cell.magnification != NAVIGATION_MAGNIFICATIONS[0]:
            raise ValueError("Start cell must be at 0.625x")
        if not self._cell.tissue_valid:
            raise ValueError("Start cell must be tissue-valid")
        self._visits = np.zeros((self.overview_size, self.overview_size), dtype=np.uint32)
        self._viewport = self._read_augmented_viewport(self._cell)
        self._evidence.clear()
        self._actions_taken = 0
        self._terminated = False
        self.belief_state.reset()
        self._visited_cell_ids = {self._cell.cell_id}
        self._revisit_count = 0
        self._magnification_histogram = Counter({f"{self._cell.magnification:g}": 1})
        self._trajectory = [self._trajectory_point(action=None)]
        increment_visits(
            self._visits,
            self._canonical_observation_transform,
            self._cell.bounds,
        )
        info = self._info()
        info["reset_visit"] = True
        return self._observation(), info

    def action_mask(self) -> np.ndarray:
        return self._geometric_orientation.mask_from_canonical(self._canonical_action_mask())

    def _canonical_action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        if self._terminated:
            return mask
        assert self._cell is not None
        if self.remaining_actions <= 1:
            mask[Action.DIAGNOSE_CANCEROUS] = True
            mask[Action.DIAGNOSE_BENIGN] = True
            return mask
        # Pan targets are the precomputed nearest tissue-valid cells in each
        # cardinal direction. Child IDs remain the four geometric quadrants,
        # so tissue validity is enforced here rather than rewriting topology.
        for index, cell_id in enumerate(self._cell.neighbor_ids):
            mask[index] = self._is_tissue_valid_target(cell_id)
        for index, cell_id in enumerate(self._cell.child_ids, start=4):
            mask[index] = self._is_tissue_valid_target(cell_id)
        # Zoom-out keeps the existing geometric parent transition. A valid
        # child produced from the tissue graph necessarily has a valid parent.
        mask[Action.ZOOM_OUT] = self._cell.parent_id >= 0
        if self.evidence_count >= self.min_diagnostic_evidence:
            mask[Action.DIAGNOSE_CANCEROUS] = True
            mask[Action.DIAGNOSE_BENIGN] = True
        return mask

    def _is_tissue_valid_target(self, cell_id: int) -> bool:
        """Return whether a graph edge resolves to a navigable tissue cell."""

        if cell_id < 0:
            return False
        assert self._slide is not None
        return bool(self._slide.cells[cell_id].tissue_valid)

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
        evidence_count_before = self.evidence_count
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
        canonical_selected = self._geometric_orientation.to_canonical_action(selected)
        self._cell = self._resolve_navigation(canonical_selected)
        revisited = self._cell.cell_id in self._visited_cell_ids
        if revisited:
            self._revisit_count += 1
        self._visited_cell_ids.add(self._cell.cell_id)
        self._magnification_histogram[f"{self._cell.magnification:g}"] += 1
        self._viewport = self._read_augmented_viewport(self._cell)
        assert self._observation_transform is not None
        assert self._canonical_observation_transform is not None
        increment_visits(
            self._visits,
            self._canonical_observation_transform,
            self._cell.bounds,
        )
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
            evidence_added=evidence_added,
            evidence_count_before=evidence_count_before,
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
        assert self._observation_transform is not None and self._spatial_overview is not None
        assert self._canonical_observation_transform is not None
        assert self._visits is not None and self._viewport is not None
        overview = rgb_chw_float(self._spatial_overview)
        agent_bounds = self._agent_bounds(self._cell.bounds)
        location = location_mask(self._canonical_observation_transform, self._cell.bounds)
        visits = np.minimum(self._visits, self.max_visit_count).astype(np.float32)
        if self.geometric_augmentation:
            location = self._geometric_orientation.apply_image(location)
            visits = self._geometric_orientation.apply_image(visits)
        visits /= float(self.max_visit_count)
        magnification_index = NAVIGATION_MAGNIFICATIONS.index(self._cell.magnification)
        center_x = (agent_bounds[0] + agent_bounds[2]) / 2
        center_y = (agent_bounds[1] + agent_bounds[3]) / 2
        roi_x, roi_y = self._observation_transform.global_to_normalized_point(
            center_x,
            center_y,
        )
        scalars = np.asarray(
            [
                magnification_index / (len(NAVIGATION_MAGNIFICATIONS) - 1),
                self.remaining_actions / self.max_episode_steps,
                float(self.belief_state.probability),
                roi_x,
                roi_y,
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
        assert self._observation_transform is not None
        assert self._augmentation_frame_size is not None
        assert self._canonical_navigation_roi is not None
        observed_views = sum(self._magnification_histogram.values())
        agent_bounds = self._agent_bounds(self._cell.bounds)
        row, column = self._agent_row_column(self._cell)
        center_x = (agent_bounds[0] + agent_bounds[2]) / 2
        center_y = (agent_bounds[1] + agent_bounds[3]) / 2
        roi_x, roi_y = self._observation_transform.global_to_normalized_point(
            center_x,
            center_y,
        )
        augmented_frame_size = self._geometric_orientation.transformed_size(
            *self._augmentation_frame_size
        )
        return {
            "slide_id": self._slide.slide_id,
            "label": self._slide.label,
            "cell_id": self._cell.cell_id,
            "start_position": self.start_position,
            "geometric_augmentation": self.geometric_augmentation,
            "geometric_orientation": self._geometric_orientation.name,
            "photometric_augmentation": self.photometric_augmentation,
            "hematoxylin_scale": self._stain_augmentation.hematoxylin_scale,
            "eosin_scale": self._stain_augmentation.eosin_scale,
            "augmentation_frame_size_level0": augmented_frame_size,
            "canonical_augmentation_frame_size_level0": self._augmentation_frame_size,
            "magnification": self._cell.magnification,
            "row": row,
            "column": column,
            "cell_bounds_level0": agent_bounds,
            "canonical_row": self._cell.row,
            "canonical_column": self._cell.column,
            "canonical_cell_bounds_level0": self._cell.bounds,
            "navigation_roi_level0": self._observation_transform.global_bounds,
            "canonical_navigation_roi_level0": self._canonical_navigation_roi,
            "roi_x": float(np.clip(roi_x, 0.0, 1.0)),
            "roi_y": float(np.clip(roi_y, 0.0, 1.0)),
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
        assert self._augmentation_frame_size is not None
        bounds = self._agent_bounds(self._cell.bounds)
        row, column = self._agent_row_column(self._cell)
        return {
            "step": self._actions_taken,
            "action": action,
            "cell_id": self._cell.cell_id,
            "magnification": self._cell.magnification,
            "row": row,
            "column": column,
            "bounds": list(bounds),
            "canonical_row": self._cell.row,
            "canonical_column": self._cell.column,
            "canonical_bounds": list(self._cell.bounds),
            "geometric_orientation": self._geometric_orientation.name,
            "canonical_augmentation_frame_size_level0": list(self._augmentation_frame_size),
            "belief": float(self.belief_state.probability),
            "evidence_count": self.evidence_count,
        }

    @property
    def trajectory(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._trajectory)

    def render(self) -> np.ndarray:
        from wsi_lab.envs.rendering import render_state

        if (
            self._slide is None
            or self._cell is None
            or self._canonical_observation_transform is None
            or self._canonical_spatial_overview is None
            or self._observation_transform is None
            or self._spatial_overview is None
            or self._visits is None
        ):
            raise RuntimeError("reset() must be called before render()")
        canonical = render_state(
            self._canonical_spatial_overview,
            self._canonical_observation_transform,
            self._cell.bounds,
            self._visits,
            self.max_visit_count,
        )
        return (
            self._geometric_orientation.apply_image(canonical)
            if self.geometric_augmentation
            else canonical
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
