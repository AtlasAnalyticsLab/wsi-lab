from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from wsi_lab.envs.actions import Action
from wsi_lab.envs.wsi_env import WSIEnv
from wsi_lab.manual.controller import ACTION_KEY_BINDINGS, ManualController
from wsi_lab.manual.ui import ManualPlayWindow, observation_panels
from wsi_lab.testing.synthetic import DeterministicBelief, SyntheticEpisodeSource


class RecordingEnvironment:
    def __init__(self, environment: WSIEnv) -> None:
        self.environment = environment
        self.step_calls: list[int] = []

    def reset(self, **kwargs: Any):
        return self.environment.reset(**kwargs)

    def step(self, action: int):
        self.step_calls.append(action)
        return self.environment.step(action)

    def close(self) -> None:
        self.environment.close()


def _environment(*, minimum_evidence: int = 4) -> WSIEnv:
    return WSIEnv(
        SyntheticEpisodeSource(overview_size=32),
        DeterministicBelief(),
        split="val",
        evaluation=True,
        overview_size=32,
        viewport_size=24,
        max_episode_steps=20,
        min_diagnostic_evidence=minimum_evidence,
    )


def test_every_environment_action_has_one_keyboard_binding() -> None:
    actions = [binding.action for binding in ACTION_KEY_BINDINGS]
    keys = [binding.key for binding in ACTION_KEY_BINDINGS]

    assert set(actions) == set(Action)
    assert len(actions) == len(set(actions))
    assert len(keys) == len(set(keys))


def test_masked_keyboard_action_never_calls_environment_step() -> None:
    recording = RecordingEnvironment(_environment())
    controller = ManualController(  # type: ignore[arg-type]
        recording,
        slide_ids=("synthetic_benign", "synthetic_cancer"),
        seed=3,
    )
    controller.reset("synthetic_benign")

    assert not controller.action_mask[Action.ZOOM_OUT]
    event = controller.handle_key("-")

    assert event.kind == "blocked"
    assert event.action is Action.ZOOM_OUT
    assert recording.step_calls == []
    assert controller.info is not None
    assert controller.info["actions_taken"] == 0
    assert controller.cumulative_reward == 0.0


def test_valid_manual_action_matches_direct_environment_transition() -> None:
    controlled_environment = _environment()
    direct_environment = _environment()
    controller = ManualController(
        controlled_environment,
        slide_ids=("synthetic_benign", "synthetic_cancer"),
        seed=7,
    )
    controller.reset("synthetic_cancer")
    direct_environment.reset(seed=7, options={"slide_id": "synthetic_cancer"})

    event = controller.handle_key("7")
    direct_observation, direct_reward, direct_terminated, direct_truncated, direct_info = (
        direct_environment.step(int(Action.ZOOM_IN_TL))
    )

    assert event.kind == "action"
    assert event.action is Action.ZOOM_IN_TL
    assert event.reward == pytest.approx(direct_reward)
    assert controller.terminated is direct_terminated
    assert controller.truncated is direct_truncated
    assert controller.info is not None
    assert controller.info["cell_id"] == direct_info["cell_id"]
    assert controller.info["action_mask"].tolist() == direct_info["action_mask"].tolist()
    assert controller.observation is not None
    for key, value in direct_observation.items():
        np.testing.assert_array_equal(controller.observation[key], value)


def test_terminal_episode_blocks_further_actions_until_reset() -> None:
    recording = RecordingEnvironment(_environment(minimum_evidence=0))
    controller = ManualController(  # type: ignore[arg-type]
        recording,
        slide_ids=("synthetic_benign", "synthetic_cancer"),
    )
    controller.reset("synthetic_benign")

    diagnosis = controller.handle_key("b")
    blocked = controller.handle_key("up")

    assert diagnosis.kind == "action"
    assert controller.terminated
    assert blocked.kind == "blocked"
    assert recording.step_calls == [int(Action.DIAGNOSE_BENIGN)]


def test_reset_and_slide_selection_use_environment_reset_options() -> None:
    controller = ManualController(
        _environment(),
        slide_ids=("synthetic_benign", "synthetic_cancer"),
    )
    controller.reset("synthetic_benign")
    assert controller.current_slide_id == "synthetic_benign"

    controller.handle_key("]")
    assert controller.current_slide_id == "synthetic_cancer"
    controller.handle_key("[")
    assert controller.current_slide_id == "synthetic_benign"
    controller.handle_key("r")
    assert controller.current_slide_id == "synthetic_benign"
    assert controller.info is not None
    assert controller.info["actions_taken"] == 0

    with pytest.raises(ValueError, match="not available"):
        controller.reset("not_in_split")


def test_observation_panels_display_every_global_channel() -> None:
    environment = _environment()
    observation, _ = environment.reset(options={"slide_id": "synthetic_benign"})

    viewport, overview, composite, location, visits = observation_panels(observation)

    np.testing.assert_array_equal(viewport, observation["viewport"].transpose(1, 2, 0))
    np.testing.assert_array_equal(overview, observation["global"][:3].transpose(1, 2, 0))
    np.testing.assert_array_equal(location, observation["global"][3])
    np.testing.assert_array_equal(visits, observation["global"][4])
    assert composite.shape == overview.shape
    assert np.isfinite(composite).all()
    assert float(composite.min()) >= 0.0
    assert float(composite.max()) <= 1.0


def test_manual_window_refreshes_after_keyboard_action() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controller = ManualController(
        _environment(),
        slide_ids=("synthetic_benign", "synthetic_cancer"),
    )
    controller.reset("synthetic_benign")
    window = ManualPlayWindow(controller)

    try:
        assert len(window.figure.axes) == 7
        assert window.viewport_axis.get_title() == "Current viewport"
        assert window.location_axis.get_title() == "Current location (global 3)"
        assert window.visits_axis.get_title() == "Visited map (global 4)"
        window._on_key(SimpleNamespace(key="7"))
        assert controller.latest_action is Action.ZOOM_IN_TL
        assert controller.info is not None
        assert controller.info["actions_taken"] == 1
        assert "ZOOM_IN_TL" in window.status_axis.texts[0].get_text()
    finally:
        plt.close(window.figure)
