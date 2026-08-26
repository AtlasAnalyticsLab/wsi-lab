from __future__ import annotations

import numpy as np
import pytest

from rl4xs_wsi.envs.actions import Action
from rl4xs_wsi.envs.wsi_env import make_env
from rl4xs_wsi.testing.synthetic import DeterministicBelief, SyntheticEpisodeSource


def make_test_env(**kwargs):
    return make_env(
        source=SyntheticEpisodeSource(overview_size=32),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
        **kwargs,
    )


def test_observation_contract_and_initial_visit() -> None:
    env = make_test_env()
    observation, info = env.reset(seed=2, options={"slide_id": "synthetic_cancer"})
    assert observation["global"].shape == (5, 32, 32)
    assert observation["viewport"].shape == (3, 24, 24)
    assert observation["scalars"].tolist() == pytest.approx([0.0, 1.0, 0.5])
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
