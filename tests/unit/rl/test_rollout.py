from __future__ import annotations

import torch

from rl4xs_wsi.rl.rollout import RecurrentRolloutBuffer, compute_gae


def _observation(num_envs: int, value: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "global": torch.full((num_envs, 5, 224, 224), value),
        "viewport": torch.full((num_envs, 3, 224, 224), value),
        "scalars": torch.zeros(num_envs, 3),
    }


def test_gae_bootstraps_truncation_but_stops_trace() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[10.0], [20.0]])
    terminated = torch.tensor([[False], [True]])
    truncated = torch.tensor([[True], [False]])
    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.equal(advantages, torch.tensor([[11.0], [2.0]]))
    assert torch.equal(returns, advantages)


def test_sequence_batches_keep_temporal_order_and_initial_states() -> None:
    buffer = RecurrentRolloutBuffer(4, 2, hidden_size=3)
    masks = torch.ones(2, 11, dtype=torch.bool)
    for step in range(4):
        hidden = torch.full((1, 2, 3), float(step))
        actions = torch.tensor([step * 10, step * 10 + 1])
        # Actions only need to be in range for model replay; encode the step
        # modulo 11 while keeping the temporal signature in scalar observations.
        actions %= 11
        observation = _observation(2)
        observation["scalars"][:, 0] = step
        buffer.add(
            observation,
            hidden,
            torch.tensor([step == 0, step == 2]),
            masks,
            actions,
            torch.zeros(2),
            torch.zeros(2),
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
    buffer.finish(torch.zeros(2), gamma=0.99, gae_lambda=0.95)

    batches = list(
        buffer.iter_sequence_batches(
            sequence_length=2,
            num_minibatches=2,
            generator=torch.Generator().manual_seed(1),
        )
    )
    assert sum(batch.num_sequences for batch in batches) == 4
    for batch in batches:
        steps = batch.observations["scalars"][:, :, 0]
        assert torch.equal(steps[1] - steps[0], torch.ones_like(steps[0]))
        assert torch.equal(batch.initial_hidden_state[0, :, 0], steps[0])
        assert batch.action_masks.shape[-1] == 11
