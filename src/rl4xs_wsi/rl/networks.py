"""Multimodal recurrent actor-critic used by RL4XS WSI."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rl4xs_wsi.rl.distributions import MaskedCategorical

NUM_ACTIONS = 11
GLOBAL_CHANNELS = 5
VIEWPORT_CHANNELS = 3
GLOBAL_SHAPE = (5, 224, 224)
VIEWPORT_SHAPE = (3, 224, 224)
SCALAR_SIZE = 3


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """A GroupNorm/SiLU residual block with optional spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.normalization_1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.convolution_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.normalization_2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU()
        self.skip: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.skip(inputs)
        hidden = self.activation(self.normalization_1(self.convolution_1(inputs)))
        hidden = self.normalization_2(self.convolution_2(hidden))
        return self.activation(hidden + residual)


class SpatialEncoder(nn.Module):
    """The modest residual CNN specified for one actor image branch."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(32), 32),
            nn.SiLU(),
        )
        self.residuals = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(128, 256), nn.SiLU())

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(self.pool(self.residuals(self.stem(inputs))))


@dataclass(frozen=True)
class PolicyStep:
    """One recurrent policy evaluation or sampled action."""

    action: Tensor
    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    logits: Tensor
    hidden_state: Tensor


@dataclass(frozen=True)
class SequenceEvaluation:
    """PPO quantities evaluated over a time-major sequence batch."""

    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    logits: Tensor
    hidden_state: Tensor


class RecurrentActorCritic(nn.Module):
    """Separate image encoders followed by a one-layer 256-unit GRU."""

    def __init__(self, *, hidden_size: int = 256, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_actions != NUM_ACTIONS:
            raise ValueError(f"RL4XSWSI-v0 requires exactly {NUM_ACTIONS} actions")
        self.hidden_size = int(hidden_size)
        self.num_actions = int(num_actions)
        # These deliberately do not share weights.
        self.global_encoder = SpatialEncoder(GLOBAL_SHAPE[0])
        self.viewport_encoder = SpatialEncoder(VIEWPORT_SHAPE[0])
        self.scalar_encoder = nn.Sequential(
            nn.Linear(SCALAR_SIZE, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 64, 256),
            nn.SiLU(),
        )
        self.recurrent = nn.GRU(256, self.hidden_size, num_layers=1)
        self.actor = nn.Linear(self.hidden_size, self.num_actions)
        self.critic = nn.Linear(self.hidden_size, 1)
        self._initialize_parameters()

    @property
    def recurrent_layers(self) -> int:
        return 1

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for name, parameter in self.recurrent.named_parameters():
            if "weight" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = next(self.parameters())
        return torch.zeros(
            self.recurrent_layers,
            batch_size,
            self.hidden_size,
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    @staticmethod
    def _branch(
        observation: Mapping[str, Tensor], primary: str, alias: str | None = None
    ) -> Tensor:
        if primary in observation:
            return observation[primary]
        if alias is not None and alias in observation:
            return observation[alias]
        raise KeyError(f"observation is missing required branch {primary!r}")

    @staticmethod
    def _normalize_image(inputs: Tensor) -> Tensor:
        if inputs.dtype == torch.uint8:
            return inputs.to(dtype=torch.float32).div(255.0)
        if not torch.is_floating_point(inputs):
            raise TypeError("image observations must be uint8 or floating point")
        return inputs

    def encode(self, observation: Mapping[str, Tensor]) -> Tensor:
        """Encode a batch or a time-major sequence of Dict observations."""

        global_image = self._branch(observation, "global", "overview")
        viewport = self._branch(observation, "viewport")
        scalars = self._branch(observation, "scalars")
        sequence_input = global_image.ndim == 5
        if sequence_input:
            if viewport.ndim != 5 or scalars.ndim != 3:
                raise ValueError("sequence observations must be time-major [T,B,...]")
            time_steps, batch_size = global_image.shape[:2]
            flat_global = global_image.reshape(-1, *global_image.shape[2:])
            flat_viewport = viewport.reshape(-1, *viewport.shape[2:])
            flat_scalars = scalars.reshape(-1, scalars.shape[-1])
        elif global_image.ndim == 4:
            if viewport.ndim != 4 or scalars.ndim != 2:
                raise ValueError("step observations must be batched [B,...]")
            batch_size = global_image.shape[0]
            time_steps = 1
            flat_global = global_image
            flat_viewport = viewport
            flat_scalars = scalars
        else:
            raise ValueError("global image must have shape [B,C,H,W] or [T,B,C,H,W]")

        if flat_global.shape[1] != GLOBAL_CHANNELS:
            raise ValueError(
                f"global branch must have {GLOBAL_CHANNELS} channels, "
                f"got {tuple(flat_global.shape[1:])}"
            )
        if flat_viewport.shape[1] != VIEWPORT_CHANNELS:
            raise ValueError(
                f"viewport branch must have {VIEWPORT_CHANNELS} channels, "
                f"got {tuple(flat_viewport.shape[1:])}"
            )
        if min(flat_global.shape[-2:]) < 1 or min(flat_viewport.shape[-2:]) < 1:
            raise ValueError("spatial observation dimensions must be positive")
        if (
            flat_global.shape[0] != flat_viewport.shape[0]
            or flat_global.shape[0] != flat_scalars.shape[0]
        ):
            raise ValueError("all observation branches must share batch/time dimensions")
        if flat_scalars.shape[-1] != SCALAR_SIZE:
            raise ValueError(f"scalars branch must contain exactly {SCALAR_SIZE} values")

        parameter = next(self.parameters())
        global_float = self._normalize_image(flat_global).to(dtype=parameter.dtype)
        viewport_float = self._normalize_image(flat_viewport).to(dtype=parameter.dtype)
        scalar_float = flat_scalars.to(dtype=parameter.dtype)
        fused = self.fusion(
            torch.cat(
                (
                    self.global_encoder(global_float),
                    self.viewport_encoder(viewport_float),
                    self.scalar_encoder(scalar_float),
                ),
                dim=-1,
            )
        )
        if sequence_input:
            return fused.reshape(time_steps, batch_size, -1)
        return fused.unsqueeze(0)

    def recurrent_features(
        self,
        encoded: Tensor,
        hidden_state: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the GRU while resetting selected environments at boundaries."""

        if encoded.ndim != 3 or encoded.shape[-1] != 256:
            raise ValueError("encoded observations must have shape [T,B,256]")
        time_steps, batch_size = encoded.shape[:2]
        expected_hidden = (self.recurrent_layers, batch_size, self.hidden_size)
        if tuple(hidden_state.shape) != expected_hidden:
            raise ValueError(
                f"hidden_state must have shape {expected_hidden}, got {tuple(hidden_state.shape)}"
            )
        if episode_starts.shape != (time_steps, batch_size):
            raise ValueError(
                f"episode_starts must have shape [T,B], got {tuple(episode_starts.shape)}"
            )
        if episode_starts.dtype != torch.bool:
            raise TypeError("episode_starts must have boolean dtype")

        hidden = hidden_state
        outputs: list[Tensor] = []
        for step in range(time_steps):
            keep = (~episode_starts[step]).to(dtype=hidden.dtype).view(1, batch_size, 1)
            hidden = hidden * keep
            output, hidden = self.recurrent(encoded[step : step + 1], hidden)
            outputs.append(output)
        return torch.cat(outputs, dim=0), hidden

    def evaluate_actions_sequence(
        self,
        observation: Mapping[str, Tensor],
        hidden_state: Tensor,
        episode_starts: Tensor,
        action_mask: Tensor,
        actions: Tensor,
    ) -> SequenceEvaluation:
        encoded = self.encode(observation)
        recurrent, final_hidden = self.recurrent_features(encoded, hidden_state, episode_starts)
        logits = self.actor(recurrent)
        if action_mask.shape != logits.shape:
            raise ValueError(
                f"action_mask must have shape {tuple(logits.shape)}, got {tuple(action_mask.shape)}"
            )
        if actions.shape != logits.shape[:-1]:
            raise ValueError(
                f"actions must have shape {tuple(logits.shape[:-1])}, got {tuple(actions.shape)}"
            )
        distribution = MaskedCategorical(logits, action_mask)
        values = self.critic(recurrent).squeeze(-1)
        return SequenceEvaluation(
            log_prob=distribution.log_prob(actions.long()),
            entropy=distribution.entropy(),
            value=values,
            logits=distribution.logits,
            hidden_state=final_hidden,
        )

    def get_action_and_value(
        self,
        observation: Mapping[str, Tensor],
        hidden_state: Tensor,
        episode_starts: Tensor,
        action_mask: Tensor,
        *,
        action: Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicyStep:
        if episode_starts.ndim != 1:
            raise ValueError("step episode_starts must have shape [B]")
        if action_mask.ndim != 2:
            raise ValueError("step action_mask must have shape [B,A]")
        encoded = self.encode(observation)
        recurrent, final_hidden = self.recurrent_features(
            encoded, hidden_state, episode_starts.unsqueeze(0)
        )
        recurrent_step = recurrent.squeeze(0)
        logits = self.actor(recurrent_step)
        distribution = MaskedCategorical(logits, action_mask)
        selected_action = (
            distribution.mode
            if action is None and deterministic
            else distribution.sample()
            if action is None
            else action.long()
        )
        value = self.critic(recurrent_step).squeeze(-1)
        return PolicyStep(
            action=selected_action,
            log_prob=distribution.log_prob(selected_action),
            entropy=distribution.entropy(),
            value=value,
            logits=distribution.logits,
            hidden_state=final_hidden,
        )

    def get_value(
        self,
        observation: Mapping[str, Tensor],
        hidden_state: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return step values and the resulting recurrent state."""

        encoded = self.encode(observation)
        if episode_starts.ndim == 1:
            episode_starts = episode_starts.unsqueeze(0)
        recurrent, final_hidden = self.recurrent_features(encoded, hidden_state, episode_starts)
        values = self.critic(recurrent).squeeze(-1)
        return values.squeeze(0) if values.shape[0] == 1 else values, final_hidden


__all__ = [
    "GLOBAL_CHANNELS",
    "GLOBAL_SHAPE",
    "NUM_ACTIONS",
    "SCALAR_SIZE",
    "VIEWPORT_CHANNELS",
    "VIEWPORT_SHAPE",
    "PolicyStep",
    "RecurrentActorCritic",
    "ResidualBlock",
    "SequenceEvaluation",
    "SpatialEncoder",
]
