"""Memory-conscious rollout storage and recurrent sequence batching."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rl4xs_wsi.rl.networks import (
    GLOBAL_CHANNELS,
    GLOBAL_SHAPE,
    NUM_ACTIONS,
    SCALAR_SIZE,
    VIEWPORT_CHANNELS,
    VIEWPORT_SHAPE,
)


def _as_tensor(value: Any, *, device: torch.device | str | None = None) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device) if device is not None else value
    return torch.as_tensor(value, device=device)


def _pack_image(value: Any, expected_channels: int, name: str) -> Tensor:
    image = _as_tensor(value)
    if image.ndim != 4 or image.shape[1] != expected_channels:
        raise ValueError(
            f"{name} must have shape [B,{expected_channels},H,W], got {tuple(image.shape)}"
        )
    if min(image.shape[-2:]) < 1:
        raise ValueError(f"{name} spatial dimensions must be positive")
    if image.dtype == torch.uint8:
        return image.contiguous()
    if not torch.is_floating_point(image):
        raise TypeError(f"{name} must be uint8 or floating point")
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f"{name} contains non-finite values")
    minimum = float(image.min()) if image.numel() else 0.0
    maximum = float(image.max()) if image.numel() else 0.0
    if minimum < -1e-6:
        raise ValueError(f"{name} values must be non-negative")
    if maximum <= 1.0 + 1e-6:
        image = image.clamp(0.0, 1.0).mul(255.0)
    elif maximum <= 255.0 + 1e-6:
        image = image.clamp(0.0, 255.0)
    else:
        raise ValueError(f"{name} values must lie in [0,1] or [0,255]")
    return image.round().to(dtype=torch.uint8).contiguous()


def pack_observation(observation: Mapping[str, Any]) -> dict[str, Tensor]:
    """Quantize actor images once so collection and replay see identical input."""

    global_value = observation.get("global", observation.get("overview"))
    if global_value is None:
        raise KeyError("observation is missing 'global'")
    if "viewport" not in observation or "scalars" not in observation:
        raise KeyError("observation requires 'viewport' and 'scalars' branches")
    packed_global = _pack_image(global_value, GLOBAL_CHANNELS, "global")
    packed_viewport = _pack_image(observation["viewport"], VIEWPORT_CHANNELS, "viewport")
    scalars = _as_tensor(observation["scalars"]).to(dtype=torch.float32).contiguous()
    if scalars.ndim != 2 or scalars.shape[-1] != SCALAR_SIZE:
        raise ValueError(f"scalars must have shape [B,{SCALAR_SIZE}]")
    if scalars.shape[0] != packed_global.shape[0] or scalars.shape[0] != packed_viewport.shape[0]:
        raise ValueError("all observation branches must have the same batch size")
    if not bool(torch.isfinite(scalars).all()):
        raise ValueError("scalars contain non-finite values")
    return {
        "global": packed_global,
        "viewport": packed_viewport,
        "scalars": scalars,
    }


def observation_to(
    observation: Mapping[str, Tensor], device: torch.device | str
) -> dict[str, Tensor]:
    return {name: value.to(device=device, non_blocking=True) for name, value in observation.items()}


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE with correct Gymnasium termination/truncation semantics.

    A truncation bootstraps from its genuine final observation but stops the
    recursive advantage trace, preventing leakage into an auto-reset episode.
    A true termination neither bootstraps nor continues the trace.
    """

    tensors = (rewards, values, next_values, terminated, truncated)
    if any(tensor.shape != rewards.shape for tensor in tensors[1:]):
        raise ValueError("all GAE tensors must have identical [T,N] shapes")
    if rewards.ndim != 2:
        raise ValueError("GAE tensors must have shape [T,N]")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must be boolean")
    if bool((terminated & truncated).any()):
        raise ValueError("a transition cannot be both terminated and truncated")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must lie in [0,1]")

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        can_bootstrap = (~terminated[step]).to(dtype=rewards.dtype)
        can_continue = (~(terminated[step] | truncated[step])).to(dtype=rewards.dtype)
        delta = rewards[step] + gamma * can_bootstrap * next_values[step] - values[step]
        next_advantage = delta + gamma * gae_lambda * can_continue * next_advantage
        advantages[step] = next_advantage
    return advantages, advantages + values


@dataclass(frozen=True)
class SequenceBatch:
    """One minibatch of complete, time-major truncated-BPTT sequences."""

    observations: dict[str, Tensor]
    initial_hidden_state: Tensor
    episode_starts: Tensor
    action_masks: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    terminated: Tensor
    truncated: Tensor

    @property
    def sequence_length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_sequences(self) -> int:
        return int(self.actions.shape[1])

    @property
    def size(self) -> int:
        return self.sequence_length * self.num_sequences

    def to(self, device: torch.device | str) -> SequenceBatch:
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, dict):
                values[field.name] = observation_to(value, device)
            else:
                values[field.name] = value.to(device=device, non_blocking=True)
        return SequenceBatch(**values)


class RecurrentRolloutBuffer:
    """Time-major rollout storage retaining masks and pre-step GRU states."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        *,
        hidden_size: int = 256,
        recurrent_layers: int = 1,
        storage_device: torch.device | str = "cpu",
        global_shape: tuple[int, int, int] = GLOBAL_SHAPE,
        viewport_shape: tuple[int, int, int] = VIEWPORT_SHAPE,
    ) -> None:
        if rollout_steps <= 0 or num_envs <= 0 or hidden_size <= 0:
            raise ValueError("rollout_steps, num_envs, and hidden_size must be positive")
        if recurrent_layers != 1:
            raise ValueError("RL4XS recurrent PPO uses exactly one GRU layer")
        self.rollout_steps = int(rollout_steps)
        self.num_envs = int(num_envs)
        self.hidden_size = int(hidden_size)
        self.recurrent_layers = int(recurrent_layers)
        self.device = torch.device(storage_device)
        self.global_shape = tuple(int(value) for value in global_shape)
        self.viewport_shape = tuple(int(value) for value in viewport_shape)
        if (
            len(self.global_shape) != 3
            or self.global_shape[0] != GLOBAL_CHANNELS
            or min(self.global_shape[1:]) < 1
        ):
            raise ValueError(
                f"global_shape must be [{GLOBAL_CHANNELS},H,W], got {self.global_shape}"
            )
        if (
            len(self.viewport_shape) != 3
            or self.viewport_shape[0] != VIEWPORT_CHANNELS
            or min(self.viewport_shape[1:]) < 1
        ):
            raise ValueError(
                f"viewport_shape must be [{VIEWPORT_CHANNELS},H,W], got {self.viewport_shape}"
            )
        shape = (self.rollout_steps, self.num_envs)
        self.observations = {
            "global": torch.empty(
                *shape, *self.global_shape, dtype=torch.uint8, device=self.device
            ),
            "viewport": torch.empty(
                *shape, *self.viewport_shape, dtype=torch.uint8, device=self.device
            ),
            "scalars": torch.empty(*shape, SCALAR_SIZE, dtype=torch.float32, device=self.device),
        }
        self.hidden_states = torch.empty(
            self.rollout_steps,
            self.recurrent_layers,
            self.num_envs,
            self.hidden_size,
            dtype=torch.float32,
            device=self.device,
        )
        self.episode_starts = torch.empty(*shape, dtype=torch.bool, device=self.device)
        self.action_masks = torch.empty(*shape, NUM_ACTIONS, dtype=torch.bool, device=self.device)
        self.actions = torch.empty(*shape, dtype=torch.long, device=self.device)
        self.log_probs = torch.empty(*shape, dtype=torch.float32, device=self.device)
        self.values = torch.empty(*shape, dtype=torch.float32, device=self.device)
        self.rewards = torch.empty(*shape, dtype=torch.float32, device=self.device)
        self.terminated = torch.empty(*shape, dtype=torch.bool, device=self.device)
        self.truncated = torch.empty(*shape, dtype=torch.bool, device=self.device)
        self.truncation_values = torch.zeros(*shape, dtype=torch.float32, device=self.device)
        self.advantages = torch.empty(*shape, dtype=torch.float32, device=self.device)
        self.returns = torch.empty(*shape, dtype=torch.float32, device=self.device)
        self._cursor = 0
        self._finished = False

    @property
    def full(self) -> bool:
        return self._cursor == self.rollout_steps

    @property
    def size(self) -> int:
        return self.rollout_steps * self.num_envs

    def reset(self) -> None:
        self._cursor = 0
        self._finished = False

    def add(
        self,
        observation: Mapping[str, Any],
        hidden_state: Tensor,
        episode_starts: Tensor,
        action_masks: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        rewards: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        *,
        truncation_values: Tensor | None = None,
    ) -> None:
        if self.full:
            raise IndexError("rollout buffer is already full")
        packed = pack_observation(observation)
        if packed["global"].shape[0] != self.num_envs:
            raise ValueError("observation batch does not match num_envs")
        if tuple(packed["global"].shape[1:]) != self.global_shape:
            raise ValueError(
                f"global observation shape changed within rollout: "
                f"{tuple(packed['global'].shape[1:])} != {self.global_shape}"
            )
        if tuple(packed["viewport"].shape[1:]) != self.viewport_shape:
            raise ValueError(
                f"viewport observation shape changed within rollout: "
                f"{tuple(packed['viewport'].shape[1:])} != {self.viewport_shape}"
            )
        expected_hidden = (
            self.recurrent_layers,
            self.num_envs,
            self.hidden_size,
        )
        if tuple(hidden_state.shape) != expected_hidden:
            raise ValueError(f"hidden_state must have shape {expected_hidden}")
        expected_vector = (self.num_envs,)
        vector_fields = {
            "episode_starts": episode_starts,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
        }
        for name, value in vector_fields.items():
            if tuple(value.shape) != expected_vector:
                raise ValueError(f"{name} must have shape {expected_vector}")
        if tuple(action_masks.shape) != (self.num_envs, NUM_ACTIONS):
            raise ValueError(f"action_masks must have shape [{self.num_envs},{NUM_ACTIONS}]")
        if episode_starts.dtype != torch.bool:
            raise TypeError("episode_starts must be boolean")
        if action_masks.dtype != torch.bool:
            raise TypeError("action_masks must be boolean")
        if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must be boolean")
        if bool((terminated & truncated).any()):
            raise ValueError("a transition cannot be both terminated and truncated")
        if bool(truncated.any()) and truncation_values is None:
            raise ValueError("truncated transitions require final-observation values")
        if truncation_values is not None and tuple(truncation_values.shape) != expected_vector:
            raise ValueError(f"truncation_values must have shape {expected_vector}")

        step = self._cursor
        for name, value in packed.items():
            self.observations[name][step].copy_(value.to(self.device))
        self.hidden_states[step].copy_(hidden_state.detach().to(self.device))
        self.episode_starts[step].copy_(episode_starts.to(self.device))
        self.action_masks[step].copy_(action_masks.to(self.device))
        self.actions[step].copy_(actions.detach().to(self.device))
        self.log_probs[step].copy_(log_probs.detach().to(self.device))
        self.values[step].copy_(values.detach().to(self.device))
        self.rewards[step].copy_(rewards.to(self.device))
        self.terminated[step].copy_(terminated.to(self.device))
        self.truncated[step].copy_(truncated.to(self.device))
        if truncation_values is not None:
            self.truncation_values[step].copy_(truncation_values.detach().to(self.device))
        else:
            self.truncation_values[step].zero_()
        self._cursor += 1

    def finish(self, last_value: Tensor, *, gamma: float, gae_lambda: float) -> None:
        if not self.full:
            raise RuntimeError("cannot finish an incomplete rollout")
        if tuple(last_value.shape) != (self.num_envs,):
            raise ValueError(f"last_value must have shape [{self.num_envs}]")
        next_values = torch.empty_like(self.values)
        next_values[:-1].copy_(self.values[1:])
        next_values[-1].copy_(last_value.detach().to(self.device))
        next_values = torch.where(self.truncated, self.truncation_values, next_values)
        self.advantages[:], self.returns[:] = compute_gae(
            self.rewards,
            self.values,
            next_values,
            self.terminated,
            self.truncated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self._finished = True

    def _batch(self, references: list[tuple[int, int]], sequence_length: int) -> SequenceBatch:
        starts = [start for start, _ in references]
        envs = [env for _, env in references]

        def sequences(tensor: Tensor) -> Tensor:
            return torch.stack(
                [
                    tensor[start : start + sequence_length, env]
                    for start, env in zip(starts, envs, strict=True)
                ],
                dim=1,
            )

        observations = {name: sequences(value) for name, value in self.observations.items()}
        initial_hidden = torch.stack(
            [self.hidden_states[start, :, env] for start, env in zip(starts, envs, strict=True)],
            dim=1,
        )
        return SequenceBatch(
            observations=observations,
            initial_hidden_state=initial_hidden,
            episode_starts=sequences(self.episode_starts),
            action_masks=sequences(self.action_masks),
            actions=sequences(self.actions),
            old_log_probs=sequences(self.log_probs),
            old_values=sequences(self.values),
            advantages=sequences(self.advantages),
            returns=sequences(self.returns),
            terminated=sequences(self.terminated),
            truncated=sequences(self.truncated),
        )

    def iter_sequence_batches(
        self,
        sequence_length: int,
        num_minibatches: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Iterator[SequenceBatch]:
        """Shuffle whole sequences while retaining temporal order within each."""

        if not self._finished:
            raise RuntimeError("finish() must be called before sampling rollout batches")
        if sequence_length <= 0 or self.rollout_steps % sequence_length != 0:
            raise ValueError("sequence_length must divide rollout_steps exactly")
        references = [
            (start, env)
            for env in range(self.num_envs)
            for start in range(0, self.rollout_steps, sequence_length)
        ]
        if num_minibatches <= 0 or num_minibatches > len(references):
            raise ValueError("num_minibatches must be between 1 and the sequence count")
        order = torch.randperm(len(references), generator=generator).tolist()
        shuffled = [references[index] for index in order]
        chunks = np.array_split(np.arange(len(shuffled)), num_minibatches)
        for positions in chunks:
            if positions.size:
                yield self._batch(
                    [shuffled[int(position)] for position in positions], sequence_length
                )


__all__ = [
    "RecurrentRolloutBuffer",
    "SequenceBatch",
    "compute_gae",
    "observation_to",
    "pack_observation",
]
