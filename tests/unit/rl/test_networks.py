from __future__ import annotations

import torch

from rl4xs_wsi.rl.networks import RecurrentActorCritic


def _observation(batch: int) -> dict[str, torch.Tensor]:
    return {
        "global": torch.randint(0, 256, (batch, 5, 32, 32), dtype=torch.uint8),
        "viewport": torch.randint(0, 256, (batch, 3, 32, 32), dtype=torch.uint8),
        "scalars": torch.rand(batch, 3),
    }


def test_actor_critic_shapes_and_separate_image_encoders() -> None:
    model = RecurrentActorCritic()
    assert (
        model.global_encoder.stem[0].weight.data_ptr()
        != model.viewport_encoder.stem[0].weight.data_ptr()
    )
    mask = torch.zeros(2, 11, dtype=torch.bool)
    mask[:, [0, 4, 9]] = True
    result = model.get_action_and_value(
        _observation(2),
        model.initial_state(2),
        torch.ones(2, dtype=torch.bool),
        mask,
    )
    assert result.action.shape == (2,)
    assert result.value.shape == (2,)
    assert result.hidden_state.shape == (1, 2, 256)
    assert all(mask[index, action] for index, action in enumerate(result.action))


def test_episode_start_resets_only_the_selected_gru_state() -> None:
    torch.manual_seed(3)
    model = RecurrentActorCritic()
    encoded = torch.randn(3, 2, 256)
    initial = torch.randn(1, 2, 256)
    starts = torch.tensor([[False, False], [True, False], [False, False]])
    outputs, _ = model.recurrent_features(encoded, initial, starts)

    reset_output, _ = model.recurrent(encoded[1:2, 0:1], torch.zeros(1, 1, model.hidden_size))
    assert torch.allclose(outputs[1, 0], reset_output[0, 0], atol=1e-6)

    continued_output, _ = model.recurrent(
        encoded[1:2, 1:2],
        model.recurrent(encoded[0:1, 1:2], initial[:, 1:2])[1],
    )
    assert torch.allclose(outputs[1, 1], continued_output[0, 0], atol=1e-6)


def test_sequence_evaluation_preserves_time_and_batch_dimensions() -> None:
    model = RecurrentActorCritic()
    observation = {
        "global": torch.zeros(2, 1, 5, 32, 32, dtype=torch.uint8),
        "viewport": torch.zeros(2, 1, 3, 32, 32, dtype=torch.uint8),
        "scalars": torch.zeros(2, 1, 3),
    }
    masks = torch.zeros(2, 1, 11, dtype=torch.bool)
    masks[..., 4] = True
    result = model.evaluate_actions_sequence(
        observation,
        model.initial_state(1),
        torch.tensor([[True], [False]]),
        masks,
        torch.full((2, 1), 4),
    )
    assert result.log_prob.shape == (2, 1)
    assert result.value.shape == (2, 1)
    assert torch.allclose(result.log_prob, torch.zeros_like(result.log_prob))
