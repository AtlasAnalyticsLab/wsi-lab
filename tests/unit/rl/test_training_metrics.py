from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import wsi_lab.rl.trainer as trainer_module
from wsi_lab.rl.checkpoints import inspect_ppo_checkpoint, save_ppo_checkpoint
from wsi_lab.rl.networks import NUM_ACTIONS
from wsi_lab.rl.recurrent_ppo import PPOConfig, PPOUpdateMetrics
from wsi_lab.rl.trainer import PPOTrainer


def _observation(num_envs: int) -> dict[str, np.ndarray]:
    return {
        "global": np.zeros((num_envs, 5, 4, 4), dtype=np.float32),
        "viewport": np.zeros((num_envs, 3, 4, 4), dtype=np.float32),
        "scalars": np.zeros((num_envs, 5), dtype=np.float32),
    }


class _ScriptedVectorEnv:
    num_envs = 2

    def __init__(self) -> None:
        # Environment 0 spans the rollout boundary (return 1+2+3, length 3).
        # Environment 1 completes twice (returns 10+20 and 30+40).
        self._transitions = (
            ([1.0, 10.0], [False, False]),
            ([2.0, 20.0], [False, True]),
            ([3.0, 30.0], [True, False]),
            ([4.0, 40.0], [False, True]),
        )
        self._cursor = 0

    @staticmethod
    def _infos(done: np.ndarray) -> dict[str, Any]:
        final_info = np.empty(2, dtype=object)
        final_info[:] = None
        for index in np.flatnonzero(done):
            final_info[int(index)] = {"slide_id": f"slide-{int(index)}"}
        return {
            "action_mask": np.ones((2, NUM_ACTIONS), dtype=np.bool_),
            "final_info": final_info,
            "_final_info": done,
        }

    def reset(self, *, seed: int | None = None):
        del seed
        self._cursor = 0
        done = np.zeros(2, dtype=np.bool_)
        return _observation(2), self._infos(done)

    def step(self, actions: np.ndarray):
        assert actions.shape == (2,)
        rewards, terminated = self._transitions[self._cursor]
        self._cursor += 1
        terminated_array = np.asarray(terminated, dtype=np.bool_)
        return (
            _observation(2),
            np.asarray(rewards, dtype=np.float32),
            terminated_array,
            np.zeros(2, dtype=np.bool_),
            self._infos(terminated_array),
        )

    def close(self) -> None:
        pass


class _StubPolicy(nn.Module):
    hidden_size = 2
    recurrent_layers = 1

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(
            1,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype or torch.float32,
        )

    def get_action_and_value(
        self,
        observation,
        hidden_state,
        episode_starts,
        action_masks,
    ):
        del observation, episode_starts
        batch_size = action_masks.shape[0]
        zeros = torch.zeros(batch_size, device=hidden_state.device)
        return SimpleNamespace(
            action=torch.zeros(batch_size, dtype=torch.long, device=hidden_state.device),
            log_prob=zeros,
            value=zeros,
            hidden_state=hidden_state + 1.0,
        )

    def get_value(self, observation, hidden_state, episode_starts):
        del observation, episode_starts
        return torch.zeros(hidden_state.shape[1], device=hidden_state.device), hidden_state


class _Writer:
    def __init__(self) -> None:
        self.events: list[tuple[str, float, int]] = []

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.events.append((name, float(value), int(step)))


def _config() -> PPOConfig:
    return PPOConfig(
        num_envs=2,
        rollout_steps=2,
        sequence_length=1,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=8,
        checkpoint_interval=8,
        validation_interval=8,
        device="cpu",
    )


def test_episode_telemetry_tracks_returns_and_lengths_across_rollouts() -> None:
    writer = _Writer()
    trainer = PPOTrainer(_ScriptedVectorEnv(), _StubPolicy(), _config(), writer=writer)

    first = trainer.collect_rollout()

    assert [episode["episode_return"] for episode in first.completed_episodes] == [30.0]
    assert [episode["episode_length"] for episode in first.completed_episodes] == [2]
    assert trainer.completed_episode_count == 1

    second = trainer.collect_rollout()

    assert [episode["episode_return"] for episode in second.completed_episodes] == [6.0, 70.0]
    assert [episode["episode_length"] for episode in second.completed_episodes] == [3, 2]
    assert [episode["completed_episode_count"] for episode in second.completed_episodes] == [2, 3]
    assert trainer.completed_episode_count == 3

    episode_returns = [event for event in writer.events if event[0] == "training/episode_return"]
    episode_lengths = [event for event in writer.events if event[0] == "training/episode_length"]
    episode_counts = [event for event in writer.events if event[0] == "training/completed_episodes"]
    assert episode_returns == [
        ("training/episode_return", 30.0, 4),
        ("training/episode_return", 6.0, 6),
        ("training/episode_return", 70.0, 8),
    ]
    assert episode_lengths == [
        ("training/episode_length", 2.0, 4),
        ("training/episode_length", 3.0, 6),
        ("training/episode_length", 2.0, 8),
    ]
    assert episode_counts == [
        ("training/completed_episodes", 1.0, 4),
        ("training/completed_episodes", 3.0, 8),
    ]


def test_update_logs_current_learning_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _Writer()
    trainer = PPOTrainer(_ScriptedVectorEnv(), _StubPolicy(), _config(), writer=writer)
    rollout = trainer.collect_rollout().buffer
    trainer.optimizer.param_groups[0]["lr"] = 1.25e-4
    expected = PPOUpdateMetrics(
        policy_loss=1.0,
        value_loss=2.0,
        entropy=3.0,
        total_loss=4.0,
        old_approx_kl=5.0,
        approx_kl=6.0,
        clip_fraction=0.1,
        gradient_norm=0.2,
        explained_variance=None,
        epochs_completed=1,
        minibatches_completed=1,
        early_stopped=False,
    )
    monkeypatch.setattr(trainer_module, "ppo_update", lambda *args: expected)

    assert trainer.update(rollout) is expected
    assert ("training/learning_rate", 1.25e-4, 4) in writer.events


def test_completed_episode_count_round_trips_through_checkpoint(tmp_path) -> None:
    config = _config()
    model = _StubPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint = save_ppo_checkpoint(
        tmp_path / "source.pt",
        model,
        optimizer,
        config,
        global_step=8,
        update=1,
        extra={"completed_episode_count": 17},
    )
    trainer = PPOTrainer(_ScriptedVectorEnv(), _StubPolicy(), config)
    trainer._running_episode_returns[:] = 99.0
    trainer._running_episode_lengths[:] = 42
    trainer.resume(checkpoint, restore_rng=False)

    assert trainer.global_step == 8
    assert trainer.completed_episode_count == 17
    assert not trainer._running_episode_returns.any()
    assert not trainer._running_episode_lengths.any()


def test_final_checkpoint_and_summary_include_completed_episode_count(tmp_path) -> None:
    trainer = PPOTrainer(_ScriptedVectorEnv(), _StubPolicy(), _config())
    trainer.global_step = trainer.config.total_timesteps
    trainer._completed_episode_count = 9

    summary = trainer.train(checkpoint_dir=tmp_path, resume_latest=False)

    checkpoint = inspect_ppo_checkpoint(tmp_path / "checkpoint_000000000008.pt")
    assert checkpoint.extra == {"final": True, "completed_episode_count": 9}
    assert summary.completed_episodes == 9
