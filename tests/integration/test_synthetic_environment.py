from __future__ import annotations

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from rl4xs_wsi.envs import make_env
from rl4xs_wsi.testing.synthetic import DeterministicBelief, SyntheticEpisodeSource


def test_environment_checker() -> None:
    environment = make_env(
        source=SyntheticEpisodeSource(overview_size=32),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=32,
    )
    check_env(environment, skip_render_check=False)
    environment.close()


def test_installed_gymnasium_environment_id_constructs() -> None:
    environment = gym.make(
        "RL4XSWSI-v0",
        overview_size=32,
        viewport_size=32,
        max_episode_steps=12,
    )
    try:
        observation, info = environment.reset(seed=1)
        assert set(observation) == {"global", "viewport", "scalars"}
        assert info["action_mask"].shape == (11,)
    finally:
        environment.close()
