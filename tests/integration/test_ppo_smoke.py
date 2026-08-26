from __future__ import annotations

import math

from wsi_lab.envs import make_env
from wsi_lab.rl.networks import RecurrentActorCritic
from wsi_lab.rl.recurrent_ppo import PPOConfig
from wsi_lab.rl.trainer import PPOTrainer, make_vector_env
from wsi_lab.testing.synthetic import DeterministicBelief, SyntheticEpisodeSource


def _factory():
    return make_env(
        source=SyntheticEpisodeSource(overview_size=32),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=32,
        max_episode_steps=3,
        min_diagnostic_evidence=0,
    )


def test_synthetic_rollout_and_single_optimizer_step(tmp_path) -> None:
    vector_env = make_vector_env([_factory, _factory], backend="sync")
    config = PPOConfig(
        seed=4,
        num_envs=2,
        rollout_steps=4,
        sequence_length=2,
        num_minibatches=2,
        update_epochs=1,
        total_timesteps=8,
        checkpoint_interval=8,
        validation_interval=8,
        device="cpu",
    )
    trainer = PPOTrainer(vector_env, RecurrentActorCritic(), config)
    try:
        summary = trainer.train(checkpoint_dir=tmp_path, resume_latest=False)
        assert summary.global_step == 8
        assert summary.last_metrics is not None
        assert math.isfinite(summary.last_metrics.total_loss)
        assert summary.last_metrics.minibatches_completed == 2
        assert (tmp_path / "checkpoint_000000000008.pt").is_file()
    finally:
        trainer.close()
