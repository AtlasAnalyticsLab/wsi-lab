from __future__ import annotations

import numpy as np
import pytest

from wsi_lab.envs.actions import Action
from wsi_lab.envs.wsi_env import LoadedSlide, NavigationCell, make_env
from wsi_lab.testing.synthetic import DeterministicBelief, SyntheticEpisodeSource
from wsi_lab.wsi.reader import OverviewTransform


class TissueGraphSource:
    """Small graph whose right pan jumps over a disconnected background cell."""

    def __init__(self) -> None:
        overview = np.zeros((32, 32, 3), dtype=np.uint8)
        transform = OverviewTransform.fit((49152, 16384), (32, 32))
        cells = {
            0: NavigationCell(
                cell_id=0,
                magnification=0.625,
                row=0,
                column=0,
                bounds=(0, 0, 16384, 16384),
                child_ids=(10, 11, -1, -1),
                # ID 2 is the next tissue location; ID 1 is background.
                neighbor_ids=(-1, -1, -1, 2),
                tissue_fraction=1.0,
                tissue_valid=True,
            ),
            1: NavigationCell(
                cell_id=1,
                magnification=0.625,
                row=0,
                column=1,
                bounds=(16384, 0, 32768, 16384),
                tissue_fraction=0.0,
                tissue_valid=False,
            ),
            2: NavigationCell(
                cell_id=2,
                magnification=0.625,
                row=0,
                column=2,
                bounds=(32768, 0, 49152, 16384),
                neighbor_ids=(-1, -1, 0, -1),
                tissue_fraction=1.0,
                tissue_valid=True,
            ),
            10: NavigationCell(
                cell_id=10,
                magnification=1.25,
                row=0,
                column=0,
                bounds=(0, 0, 8192, 8192),
                parent_id=0,
                tissue_fraction=0.4,
                embedding_index=-1,
                tissue_valid=True,
            ),
            11: NavigationCell(
                cell_id=11,
                magnification=1.25,
                row=0,
                column=1,
                bounds=(8192, 0, 16384, 8192),
                parent_id=0,
                tissue_fraction=0.0,
                embedding_index=-1,
                tissue_valid=False,
            ),
        }
        self.slide = LoadedSlide("tissue_graph", 1, overview, transform, cells, 0)

    def slide_ids(self, split: str):
        del split
        return (self.slide.slide_id,)

    def load_slide(self, slide_id: str) -> LoadedSlide:
        assert slide_id == self.slide.slide_id
        return self.slide

    def read_viewport(self, slide_id: str, cell: NavigationCell, output_size: int) -> np.ndarray:
        del slide_id, cell
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
        raise AssertionError((slide_id, cell))


def make_test_env(**kwargs):
    return make_env(
        source=SyntheticEpisodeSource(overview_size=32),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
        **kwargs,
    )


def advance_to_first_evidence(env):
    info = None
    reward = 0.0
    for action in (
        Action.PAN_UP,
        Action.PAN_LEFT,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
    ):
        _, reward, _, _, info = env.step(action)
    assert info is not None
    return reward, info


def test_observation_contract_and_initial_visit() -> None:
    env = make_test_env()
    observation, info = env.reset(seed=2, options={"slide_id": "synthetic_cancer"})
    assert observation["global"].shape == (5, 32, 32)
    assert observation["viewport"].shape == (3, 24, 24)
    assert observation["scalars"].tolist() == pytest.approx([0.0, 1.0, 0.5, 0.75, 0.75])
    assert observation["global"][4].max() == pytest.approx(0.1)
    assert info["evidence_count"] == 0
    assert info["action_mask"].dtype == np.bool_
    env.close()


def test_invalid_action_is_noop_without_budget_or_visit() -> None:
    env = make_test_env()
    before, info = env.reset(seed=1, options={"slide_id": "synthetic_benign"})
    assert not info["action_mask"][Action.PAN_DOWN]
    after, reward, terminated, truncated, next_info = env.step(Action.PAN_DOWN)
    assert reward == pytest.approx(-0.001)
    assert not terminated and not truncated
    assert next_info["invalid_action"]
    assert next_info["actions_taken"] == 0
    np.testing.assert_array_equal(after["global"], before["global"])


def test_sparse_embedding_child_remains_navigable_without_belief_update() -> None:
    env = make_test_env(min_diagnostic_evidence=1)
    _, info = env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    for action in (
        Action.PAN_UP,
        Action.PAN_LEFT,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
    ):
        _, _, _, _, info = env.step(action)
    assert info["magnification"] == 5.0
    before_belief = info["belief"]
    before_count = info["evidence_count"]
    assert info["action_mask"][Action.ZOOM_IN_BR]
    _, _, _, _, info = env.step(Action.ZOOM_IN_BR)
    assert info["magnification"] == 10.0
    assert info["evidence_count"] == before_count
    assert info["belief"] == before_belief


def test_tissue_graph_masks_background_and_jumps_to_disconnected_tissue() -> None:
    env = make_env(
        source=TissueGraphSource(),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
    )
    _, info = env.reset(options={"slide_id": "tissue_graph"})

    assert not info["action_mask"][Action.PAN_UP]
    assert not info["action_mask"][Action.PAN_DOWN]
    assert not info["action_mask"][Action.PAN_LEFT]
    assert info["action_mask"][Action.PAN_RIGHT]

    _, _, _, _, info = env.step(Action.PAN_RIGHT)
    assert info["cell_id"] == 2
    assert info["column"] == 2


def test_zoom_masks_non_tissue_child_but_missing_embedding_tissue_is_navigable() -> None:
    env = make_env(
        source=TissueGraphSource(),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
    )
    _, info = env.reset(options={"slide_id": "tissue_graph"})

    assert info["action_mask"][Action.ZOOM_IN_TL]
    assert not info["action_mask"][Action.ZOOM_IN_TR]
    _, reward, _, _, info = env.step(Action.ZOOM_IN_TL)
    assert info["cell_id"] == 10
    assert info["evidence_count"] == 0
    assert reward == pytest.approx(-0.001)
    assert info["action_mask"][Action.ZOOM_OUT]


def test_final_action_is_reserved_for_diagnosis() -> None:
    env = make_test_env(max_episode_steps=2, min_diagnostic_evidence=99)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    _, _, terminated, _, info = env.step(Action.PAN_UP)
    assert not terminated
    assert not info["action_mask"][:9].any()
    assert info["action_mask"][9:].all()
    _, reward, terminated, truncated, info = env.step(Action.DIAGNOSE_CANCEROUS)
    assert terminated and not truncated
    assert reward == pytest.approx(0.999)
    assert not info["action_mask"].any()


def test_cross_scale_evidence_is_unique_per_cell_and_scale() -> None:
    env = make_test_env(min_diagnostic_evidence=1)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    env.step(Action.PAN_UP)
    env.step(Action.PAN_LEFT)
    env.step(Action.ZOOM_IN_TL)
    _, _, _, _, info = env.step(Action.ZOOM_IN_TL)
    assert info["evidence_count"] == 1
    _, _, _, _, info = env.step(Action.ZOOM_OUT)
    _, _, _, _, info = env.step(Action.ZOOM_IN_TL)
    assert info["evidence_count"] == 1
    _, _, _, _, info = env.step(Action.ZOOM_IN_TL)
    assert info["evidence_count"] == 2


def test_reset_clears_belief_before_recording_trajectory() -> None:
    env = make_test_env()
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    env.step(Action.PAN_UP)
    env.step(Action.PAN_LEFT)
    env.step(Action.ZOOM_IN_TL)
    env.step(Action.ZOOM_IN_TL)
    assert env.unwrapped.belief_state.probability != pytest.approx(0.5)
    env.reset(seed=2, options={"slide_id": "synthetic_benign"})
    assert env.unwrapped.trajectory[0]["belief"] == pytest.approx(0.5)


def test_first_evidence_updates_belief_without_belief_shaping() -> None:
    env = make_test_env(reward_type="belief", belief_reward_beta=1.0)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})

    reward, info = advance_to_first_evidence(env)

    assert info["evidence_added"]
    assert info["evidence_count"] == 1
    assert info["belief"] > 0.5
    assert reward == pytest.approx(-0.001)


def test_belief_shaping_begins_with_second_evidence() -> None:
    env = make_test_env(reward_type="belief", belief_reward_beta=1.0)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    _, first_info = advance_to_first_evidence(env)

    _, reward, _, _, second_info = env.step(Action.ZOOM_IN_TL)

    assert second_info["evidence_added"]
    assert second_info["evidence_count"] == 2
    assert reward == pytest.approx(-0.001 + second_info["belief"] - first_info["belief"])


def test_default_navigation_reward_unchanged_for_first_and_second_evidence() -> None:
    env = make_test_env(reward_type="default", belief_reward_beta=10.0)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})

    first_reward, first_info = advance_to_first_evidence(env)
    _, second_reward, _, _, second_info = env.step(Action.ZOOM_IN_TL)

    assert first_info["evidence_added"] and second_info["evidence_added"]
    assert first_reward == pytest.approx(-0.001)
    assert second_reward == pytest.approx(-0.001)
