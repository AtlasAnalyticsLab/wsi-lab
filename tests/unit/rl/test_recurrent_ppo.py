from __future__ import annotations

import math

import torch

from wsi_lab.rl.networks import SCALAR_SIZE, RecurrentActorCritic
from wsi_lab.rl.recurrent_ppo import PPOConfig, ppo_update
from wsi_lab.rl.rollout import RecurrentRolloutBuffer


def test_one_synthetic_recurrent_ppo_optimizer_step() -> None:
    torch.manual_seed(9)

    class RecordingActorCritic(RecurrentActorCritic):
        def __init__(self) -> None:
            super().__init__()
            self.replayed_masks: list[torch.Tensor] = []

        def evaluate_actions_sequence(
            self, observation, hidden_state, episode_starts, action_mask, actions
        ):
            self.replayed_masks.append(action_mask.detach().cpu().clone())
            return super().evaluate_actions_sequence(
                observation, hidden_state, episode_starts, action_mask, actions
            )

    model = RecordingActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    buffer = RecurrentRolloutBuffer(4, 2, global_shape=(5, 32, 32), viewport_shape=(3, 32, 32))
    hidden = model.initial_state(2)
    episode_starts = torch.ones(2, dtype=torch.bool)
    for step in range(4):
        observation = {
            "global": torch.randint(0, 256, (2, 5, 32, 32), dtype=torch.uint8),
            "viewport": torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8),
            "scalars": torch.rand(2, SCALAR_SIZE),
        }
        masks = torch.zeros(2, 11, dtype=torch.bool)
        masks[:, [step % 9, 9, 10]] = True
        with torch.no_grad():
            policy = model.get_action_and_value(observation, hidden, episode_starts, masks)
        terminated = torch.tensor([step == 3, False])
        buffer.add(
            observation,
            hidden,
            episode_starts,
            masks,
            policy.action,
            policy.log_prob,
            policy.value,
            torch.tensor([0.1, -0.1]),
            terminated,
            torch.zeros(2, dtype=torch.bool),
        )
        hidden = policy.hidden_state * (~terminated).view(1, 2, 1)
        episode_starts = terminated
    buffer.finish(torch.zeros(2), gamma=0.99, gae_lambda=0.95)
    before = model.actor.weight.detach().clone()
    config = PPOConfig(
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
    metrics = ppo_update(model, optimizer, buffer, config)
    assert metrics.minibatches_completed == 2
    assert math.isfinite(metrics.total_loss)
    assert not torch.equal(before, model.actor.weight)
    replayed = torch.cat([mask.reshape(-1, 11) for mask in model.replayed_masks], dim=0)
    expected_signatures = sorted(map(tuple, buffer.action_masks.reshape(-1, 11).tolist()))
    replayed_signatures = sorted(map(tuple, replayed.tolist()))
    assert replayed_signatures == expected_signatures
