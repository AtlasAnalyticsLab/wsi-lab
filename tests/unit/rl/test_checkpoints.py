from __future__ import annotations

import pytest
import torch

from rl4xs_wsi.rl.checkpoints import load_ppo_checkpoint, save_ppo_checkpoint
from rl4xs_wsi.rl.networks import RecurrentActorCritic
from rl4xs_wsi.rl.recurrent_ppo import PPOConfig
from rl4xs_wsi.rl.trainer import PPOTrainer


def test_atomic_checkpoint_round_trip(tmp_path) -> None:
    model = RecurrentActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    expected = model.actor.weight.detach().clone()
    path = save_ppo_checkpoint(
        tmp_path / "checkpoint_000000000008.pt",
        model,
        optimizer,
        PPOConfig(),
        global_step=8,
        update=1,
        extra={"belief_fingerprint": "abc"},
    )
    with torch.no_grad():
        model.actor.weight.zero_()
    loaded = load_ppo_checkpoint(path, model, optimizer, restore_rng=False)
    assert loaded.global_step == 8
    assert loaded.update == 1
    assert loaded.extra["belief_fingerprint"] == "abc"
    assert torch.equal(model.actor.weight, expected)


def test_checkpoint_rejects_dependency_mismatch_before_loading_weights(tmp_path) -> None:
    model = RecurrentActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = save_ppo_checkpoint(
        tmp_path / "checkpoint_000000000008.pt",
        model,
        optimizer,
        PPOConfig(),
        global_step=8,
        update=1,
        dependencies={"belief_fingerprint": "old"},
    )
    with torch.no_grad():
        model.actor.weight.fill_(7.0)
    sentinel = model.actor.weight.detach().clone()
    with pytest.raises(ValueError, match="dependency mismatch"):
        load_ppo_checkpoint(
            path,
            model,
            optimizer,
            restore_rng=False,
            expected_dependencies={"belief_fingerprint": "new"},
        )
    assert torch.equal(model.actor.weight, sentinel)


def test_trainer_resumes_latest_compatible_checkpoint(tmp_path) -> None:
    class StubVectorEnv:
        num_envs = 1

        def close(self):
            pass

    config = PPOConfig(
        num_envs=1,
        rollout_steps=2,
        sequence_length=1,
        num_minibatches=1,
        total_timesteps=2,
        checkpoint_interval=2,
        validation_interval=2,
        device="cpu",
    )
    source_model = RecurrentActorCritic()
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=config.learning_rate)
    compatible = save_ppo_checkpoint(
        tmp_path / "checkpoint_000000000002.pt",
        source_model,
        source_optimizer,
        config,
        global_step=2,
        update=1,
        dependencies={"belief_fingerprint": "current"},
    )
    save_ppo_checkpoint(
        tmp_path / "checkpoint_000000000004.pt",
        source_model,
        source_optimizer,
        config,
        global_step=4,
        update=2,
        dependencies={"belief_fingerprint": "stale"},
    )
    # A truncated newest candidate must be reported and skipped, not abort the
    # compatible-resume search before the older valid payload is considered.
    (tmp_path / "checkpoint_000000000006.pt").write_bytes(b"corrupt")
    trainer = PPOTrainer(
        StubVectorEnv(),
        RecurrentActorCritic(),
        config,
        dependencies={"belief_fingerprint": "current"},
    )
    resumed = trainer.resume_latest_compatible(tmp_path)
    assert resumed == compatible
    assert trainer.global_step == 2
    assert trainer.update_number == 1
