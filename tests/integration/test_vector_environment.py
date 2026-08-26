from __future__ import annotations

import gymnasium as gym
import numpy as np

from wsi_lab.envs import make_env
from wsi_lab.testing.synthetic import SyntheticEpisodeSource


def _factory():
    return make_env(
        source=SyntheticEpisodeSource(overview_size=32),
        overview_size=32,
        viewport_size=32,
        max_episode_steps=2,
    )


def test_sync_vector_environment_same_step_autoreset() -> None:
    vector = gym.vector.SyncVectorEnv(
        [_factory, _factory], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    observation, info = vector.reset(seed=1)
    assert observation["global"].shape == (2, 5, 32, 32)
    actions = np.asarray([0, 0])
    _, _, terminated, truncated, info = vector.step(actions)
    assert not terminated.any() and not truncated.any()
    assert np.asarray(info["action_mask"])[:, :9].sum() == 0
    observation, _, terminated, truncated, info = vector.step(np.asarray([9, 10]))
    assert terminated.all() and not truncated.any()
    assert observation["scalars"][:, 1].tolist() == [1.0, 1.0]
    assert "final_info" in info
    assert np.asarray(info["action_mask"]).any(axis=1).all()
    vector.close()


def test_async_vector_environment_smoke() -> None:
    vector = gym.vector.AsyncVectorEnv(
        [_factory, _factory], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    observation, _ = vector.reset(seed=1)
    assert observation["scalars"].shape == (2, 5)
    vector.close()
