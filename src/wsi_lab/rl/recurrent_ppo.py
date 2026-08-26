"""Local CleanRL-style recurrent PPO update with truncated BPTT."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from wsi_lab.rl.networks import RecurrentActorCritic
from wsi_lab.rl.rollout import RecurrentRolloutBuffer


@dataclass(frozen=True)
class PPOConfig:
    """Complete recurrent-PPO configuration mirrored by the shared RLConfig."""

    seed: int = 1
    num_envs: int = 8
    rollout_steps: int = 128
    sequence_length: int = 32
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    total_timesteps: int = 1_000_000
    anneal_learning_rate: bool = True
    clip_value_loss: bool = True
    target_kl: float | None = None
    checkpoint_interval: int = 100_000
    validation_interval: int = 100_000
    vector_backend: str = "sync"
    device: str = "auto"

    def __post_init__(self) -> None:
        positive_integers = {
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "sequence_length": self.sequence_length,
            "update_epochs": self.update_epochs,
            "num_minibatches": self.num_minibatches,
            "total_timesteps": self.total_timesteps,
            "checkpoint_interval": self.checkpoint_interval,
            "validation_interval": self.validation_interval,
        }
        for name, integer_value in positive_integers.items():
            if isinstance(integer_value, bool) or int(integer_value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.rollout_steps % self.sequence_length != 0:
            raise ValueError("sequence_length must divide rollout_steps exactly")
        sequence_count = self.num_envs * (self.rollout_steps // self.sequence_length)
        if self.num_minibatches > sequence_count:
            raise ValueError("num_minibatches cannot exceed the rollout sequence count")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        for name in ("gamma", "gae_lambda"):
            bounded_value = float(getattr(self, name))
            if not 0.0 <= bounded_value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 < self.clip_coef < 1.0:
            raise ValueError("clip_coef must lie in (0,1)")
        for name in ("entropy_coef", "value_coef", "max_grad_norm"):
            coefficient_value = float(getattr(self, name))
            if not math.isfinite(coefficient_value) or coefficient_value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_grad_norm == 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.target_kl is not None and (
            not math.isfinite(self.target_kl) or self.target_kl <= 0.0
        ):
            raise ValueError("target_kl must be finite and positive or None")
        if self.vector_backend not in {"sync", "async"}:
            raise ValueError("vector_backend must be 'sync' or 'async'")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @classmethod
    def from_object(cls, value: Any) -> PPOConfig:
        """Construct from the shared typed RLConfig or a plain mapping."""

        if isinstance(value, cls):
            return value
        names = {field.name for field in fields(cls)}
        if isinstance(value, Mapping):
            values = {name: value[name] for name in names if name in value}
        else:
            values = {name: getattr(value, name) for name in names if hasattr(value, name)}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    old_approx_kl: float
    approx_kl: float
    clip_fraction: float
    gradient_norm: float
    explained_variance: float | None
    epochs_completed: int
    minibatches_completed: int
    early_stopped: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def annealed_learning_rate(
    initial_learning_rate: float, current_timestep: int, total_timesteps: int
) -> float:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    fraction_remaining = 1.0 - min(max(current_timestep, 0), total_timesteps) / total_timesteps
    return float(initial_learning_rate) * fraction_remaining


def _explained_variance(values: Tensor, returns: Tensor) -> float | None:
    values_np = values.detach().float().cpu().numpy().reshape(-1)
    returns_np = returns.detach().float().cpu().numpy().reshape(-1)
    variance = float(np.var(returns_np))
    if variance <= 1e-12:
        return None
    return float(1.0 - np.var(returns_np - values_np) / variance)


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"PPO produced non-finite {name}")


def ppo_update(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RecurrentRolloutBuffer,
    config: PPOConfig | Any,
    *,
    generator: torch.Generator | None = None,
) -> PPOUpdateMetrics:
    """Perform one PPO update over whole, shuffled recurrent sequences."""

    config = PPOConfig.from_object(config)
    if rollout.rollout_steps != config.rollout_steps or rollout.num_envs != config.num_envs:
        raise ValueError("rollout dimensions do not match PPO configuration")
    model_device = next(model.parameters()).device
    rows: list[dict[str, float]] = []
    epochs_completed = 0
    minibatches_completed = 0
    early_stopped = False
    model.train(True)

    for epoch in range(config.update_epochs):
        epoch_kls: list[float] = []
        for cpu_batch in rollout.iter_sequence_batches(
            config.sequence_length,
            config.num_minibatches,
            generator=generator,
        ):
            batch = cpu_batch.to(model_device)
            evaluation = model.evaluate_actions_sequence(
                batch.observations,
                batch.initial_hidden_state,
                batch.episode_starts,
                batch.action_masks,
                batch.actions,
            )
            _finite("new log probabilities", evaluation.log_prob)
            _finite("values", evaluation.value)
            _finite("entropy", evaluation.entropy)

            log_ratio = evaluation.log_prob - batch.old_log_probs
            ratio = log_ratio.exp()
            with torch.no_grad():
                old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

            advantages = batch.advantages
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )
            policy_loss_unclipped = -advantages * ratio
            policy_loss_clipped = -advantages * ratio.clamp(
                1.0 - config.clip_coef, 1.0 + config.clip_coef
            )
            policy_loss = torch.maximum(policy_loss_unclipped, policy_loss_clipped).mean()

            if config.clip_value_loss:
                value_delta = evaluation.value - batch.old_values
                clipped_value = batch.old_values + value_delta.clamp(
                    -config.clip_coef, config.clip_coef
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (evaluation.value - batch.returns).pow(2),
                        (clipped_value - batch.returns).pow(2),
                    ).mean()
                )
            else:
                value_loss = 0.5 * (evaluation.value - batch.returns).pow(2).mean()
            entropy = evaluation.entropy.mean()
            total_loss = (
                policy_loss - config.entropy_coef * entropy + config.value_coef * value_loss
            )
            _finite("loss", total_loss)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            _finite("gradient norm", torch.as_tensor(gradient_norm))
            optimizer.step()

            rows.append(
                {
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "total_loss": float(total_loss.detach()),
                    "old_approx_kl": float(old_approx_kl.detach()),
                    "approx_kl": float(approx_kl.detach()),
                    "clip_fraction": float(clip_fraction.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
            epoch_kls.append(float(approx_kl.detach()))
            minibatches_completed += 1
        epochs_completed = epoch + 1
        if (
            config.target_kl is not None
            and epoch_kls
            and float(np.mean(epoch_kls)) > config.target_kl
        ):
            early_stopped = True
            break

    if not rows:
        raise RuntimeError("PPO update received no sequence minibatches")

    def mean(name: str) -> float:
        return float(np.mean([row[name] for row in rows]))

    return PPOUpdateMetrics(
        policy_loss=mean("policy_loss"),
        value_loss=mean("value_loss"),
        entropy=mean("entropy"),
        total_loss=mean("total_loss"),
        old_approx_kl=mean("old_approx_kl"),
        approx_kl=mean("approx_kl"),
        clip_fraction=mean("clip_fraction"),
        gradient_norm=mean("gradient_norm"),
        explained_variance=_explained_variance(rollout.values, rollout.returns),
        epochs_completed=epochs_completed,
        minibatches_completed=minibatches_completed,
        early_stopped=early_stopped,
    )


__all__ = [
    "PPOConfig",
    "PPOUpdateMetrics",
    "annealed_learning_rate",
    "ppo_update",
]
