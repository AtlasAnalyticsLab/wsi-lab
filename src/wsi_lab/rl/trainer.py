"""Vectorized rollout collection and recurrent-PPO training orchestration."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from wsi_lab.rl.checkpoints import (
    checkpoint_candidates,
    load_ppo_checkpoint,
    save_ppo_checkpoint,
)
from wsi_lab.rl.networks import NUM_ACTIONS, RecurrentActorCritic
from wsi_lab.rl.recurrent_ppo import (
    PPOConfig,
    PPOUpdateMetrics,
    annealed_learning_rate,
    ppo_update,
)
from wsi_lab.rl.rollout import (
    RecurrentRolloutBuffer,
    observation_to,
    pack_observation,
)

EnvFactory = Callable[[], Any]


def _shape3(value: Any, *, name: str) -> tuple[int, int, int]:
    shape = tuple(int(dimension) for dimension in value)
    if len(shape) != 3:
        raise ValueError(f"{name} must have three dimensions, got {shape}")
    return shape


RESUME_CONFIG_FIELDS = (
    "seed",
    "num_envs",
    "rollout_steps",
    "sequence_length",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "update_epochs",
    "entropy_coef",
    "value_coef",
    "max_grad_norm",
    "num_minibatches",
    "anneal_learning_rate",
    "clip_value_loss",
    "target_kl",
    "vector_backend",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def make_vector_env(env_fns: Sequence[EnvFactory], *, backend: str = "sync") -> Any:
    """Create Gymnasium vector workers with same-step autoreset when available."""

    if not env_fns:
        raise ValueError("at least one environment factory is required")
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("Gymnasium is required for recurrent PPO") from exc
    if backend not in {"sync", "async"}:
        raise ValueError("backend must be 'sync' or 'async'")
    vector_class = gym.vector.SyncVectorEnv if backend == "sync" else gym.vector.AsyncVectorEnv
    try:
        autoreset_mode = gym.vector.AutoresetMode.SAME_STEP
        return vector_class(list(env_fns), autoreset_mode=autoreset_mode)
    except (AttributeError, TypeError):
        # Gymnasium <1.1 used same-step autoreset as its fixed behavior.
        return vector_class(list(env_fns))


def _mask_array(value: Any, num_envs: int) -> np.ndarray:
    masks = np.asarray(value, dtype=np.bool_)
    if num_envs == 1 and masks.shape == (NUM_ACTIONS,):
        masks = masks[None, :]
    if masks.shape != (num_envs, NUM_ACTIONS):
        raise ValueError(
            f"info['action_mask'] must have shape [{num_envs},{NUM_ACTIONS}], got {masks.shape}"
        )
    if not masks.any(axis=1).all():
        raise ValueError("every vector environment must expose at least one valid action")
    return masks


def extract_action_masks(infos: Any, num_envs: int) -> np.ndarray:
    """Normalize Gymnasium dict-of-arrays or list-of-dicts vector info."""

    if isinstance(infos, Mapping):
        if "action_mask" in infos:
            return _mask_array(infos["action_mask"], num_envs)
        if num_envs == 1 and "reset_info" in infos:
            reset_info = infos["reset_info"]
            if isinstance(reset_info, Mapping) and "action_mask" in reset_info:
                return _mask_array(reset_info["action_mask"], num_envs)
    if isinstance(infos, Sequence) and not isinstance(infos, (str, bytes)):
        if len(infos) != num_envs:
            raise ValueError("vector info length does not match num_envs")
        masks = []
        for index, info in enumerate(infos):
            if not isinstance(info, Mapping) or "action_mask" not in info:
                raise KeyError(f"environment {index} did not return info['action_mask']")
            masks.append(np.asarray(info["action_mask"], dtype=np.bool_))
        return _mask_array(np.stack(masks), num_envs)
    raise KeyError("environment info did not contain the required action_mask")


def _object_at(value: Any, index: int) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object:
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    if isinstance(value, Mapping):
        return {name: _object_at(branch, index) for name, branch in value.items()}
    if isinstance(value, np.ndarray):
        return value[index]
    return value


def extract_final_observation(infos: Any, index: int) -> Mapping[str, Any] | None:
    if isinstance(infos, Sequence) and not isinstance(infos, (str, bytes, Mapping)):
        info = infos[index]
        if isinstance(info, Mapping):
            value = info.get("final_observation", info.get("final_obs"))
            return value if isinstance(value, Mapping) else None
        return None
    if not isinstance(infos, Mapping):
        return None
    for key in ("final_observation", "final_obs"):
        if key in infos:
            presence = infos.get(f"_{key}")
            if presence is not None and not bool(np.asarray(presence)[index]):
                return None
            value = _object_at(infos[key], index)
            return value if isinstance(value, Mapping) else None
    return None


def _final_info_at(infos: Any, index: int) -> Mapping[str, Any] | None:
    """Return one terminal info across Gymnasium vector-info formats."""

    if isinstance(infos, Sequence) and not isinstance(infos, (str, bytes, Mapping)):
        if index >= len(infos):
            return None
        info = infos[index]
        if not isinstance(info, Mapping):
            return None
        final = info.get("final_info", info)
        return final if isinstance(final, Mapping) else None
    if not isinstance(infos, Mapping) or "final_info" not in infos:
        return None
    presence = infos.get("_final_info")
    if presence is not None and not bool(np.asarray(presence, dtype=np.bool_)[index]):
        return None
    value = _object_at(infos["final_info"], index)
    return value if isinstance(value, Mapping) else None


def _single_observation_batch(observation: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in observation.items():
        array = np.asarray(value)
        result[name] = np.expand_dims(array, axis=0)
    return result


@dataclass(frozen=True)
class RolloutCollection:
    buffer: RecurrentRolloutBuffer
    completed_episodes: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TrainingSummary:
    global_step: int
    updates: int
    last_metrics: PPOUpdateMetrics | None
    completed_episodes: int = 0


class PPOTrainer:
    """Own vector state, collect masked rollouts, and invoke local PPO updates."""

    def __init__(
        self,
        vector_env: Any,
        model: RecurrentActorCritic,
        config: PPOConfig | Any,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        writer: Any | None = None,
        storage_device: torch.device | str = "cpu",
        dependencies: Mapping[str, str] | None = None,
    ) -> None:
        self.config = PPOConfig.from_object(config)
        self.vector_env = vector_env
        self.model = model
        self.device = resolve_device(self.config.device)
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, eps=1e-5
        )
        self.writer = writer
        self.storage_device = torch.device(storage_device)
        self.dependencies = {str(key): str(value) for key, value in (dependencies or {}).items()}
        observed_num_envs = int(getattr(vector_env, "num_envs", -1))
        if observed_num_envs != self.config.num_envs:
            raise ValueError(
                f"vector env has {observed_num_envs} workers, "
                f"configured num_envs={self.config.num_envs}"
            )
        seed_everything(self.config.seed)
        self.global_step = 0
        self.update_number = 0
        self._observation: dict[str, Tensor] | None = None
        self._action_masks: Tensor | None = None
        self._hidden_state: Tensor | None = None
        self._episode_starts: Tensor | None = None
        self._global_shape: tuple[int, int, int] | None = None
        self._viewport_shape: tuple[int, int, int] | None = None
        self._running_episode_returns = np.zeros(self.config.num_envs, dtype=np.float64)
        self._running_episode_lengths = np.zeros(self.config.num_envs, dtype=np.int64)
        self._completed_episode_count = 0
        observation_space = getattr(vector_env, "single_observation_space", None)
        if observation_space is not None:
            try:
                self._global_shape = _shape3(
                    observation_space["global"].shape, name="global observation"
                )
                self._viewport_shape = _shape3(
                    observation_space["viewport"].shape, name="viewport observation"
                )
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError(
                    "vector environment observation space must contain global and viewport"
                ) from exc

    def reset(self, *, seed: int | None = None) -> None:
        effective_seed = (
            self.config.seed + self.global_step // self.config.num_envs if seed is None else seed
        )
        observation, infos = self.vector_env.reset(seed=effective_seed)
        self._observation = pack_observation(observation)
        observed_global_shape = _shape3(
            self._observation["global"].shape[1:], name="global observation"
        )
        observed_viewport_shape = _shape3(
            self._observation["viewport"].shape[1:], name="viewport observation"
        )
        if self._global_shape is not None and observed_global_shape != self._global_shape:
            raise ValueError("reset global observation does not match observation_space")
        if self._viewport_shape is not None and observed_viewport_shape != self._viewport_shape:
            raise ValueError("reset viewport observation does not match observation_space")
        self._global_shape = observed_global_shape
        self._viewport_shape = observed_viewport_shape
        masks = extract_action_masks(infos, self.config.num_envs)
        self._action_masks = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        self._hidden_state = self.model.initial_state(self.config.num_envs, device=self.device)
        self._episode_starts = torch.ones(
            self.config.num_envs, dtype=torch.bool, device=self.device
        )
        # A vector reset starts entirely new episodes.  In particular this
        # prevents partial pre-checkpoint rewards from being attributed to the
        # fresh worker episodes created after resume.
        self._running_episode_returns.fill(0.0)
        self._running_episode_lengths.fill(0)

    @property
    def completed_episode_count(self) -> int:
        """Number of training episodes completed, including restored history."""

        return self._completed_episode_count

    def _record_rollout_telemetry(self, completed_episodes: Sequence[Mapping[str, Any]]) -> None:
        if self.writer is None:
            return
        for episode in completed_episodes:
            completion_step = int(episode["completion_global_step"])
            self.writer.add_scalar(
                "training/episode_return",
                float(episode["episode_return"]),
                completion_step,
            )
            self.writer.add_scalar(
                "training/episode_length",
                int(episode["episode_length"]),
                completion_step,
            )
        # Emit this on every rollout, including rollouts with no termination,
        # so dashboards show a well-defined zero/plateau rather than no data.
        self.writer.add_scalar(
            "training/completed_episodes",
            self._completed_episode_count,
            self.global_step,
        )

    def _require_state(self) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
        if self._observation is None:
            self.reset()
        assert self._observation is not None
        assert self._action_masks is not None
        assert self._hidden_state is not None
        assert self._episode_starts is not None
        return (
            self._observation,
            self._action_masks,
            self._hidden_state,
            self._episode_starts,
        )

    def _truncation_values(
        self, infos: Any, truncated: np.ndarray, hidden_after_step: Tensor
    ) -> Tensor:
        values = torch.zeros(self.config.num_envs, device=self.device)
        for index in np.flatnonzero(truncated):
            final_observation = extract_final_observation(infos, int(index))
            if final_observation is None:
                raise RuntimeError(
                    "A truncated vector transition did not provide final_observation; "
                    "correct GAE bootstrapping is impossible"
                )
            packed = pack_observation(_single_observation_batch(final_observation))
            model_observation = observation_to(packed, self.device)
            final_hidden = hidden_after_step[:, int(index) : int(index) + 1]
            with torch.no_grad():
                final_value, _ = self.model.get_value(
                    model_observation,
                    final_hidden,
                    torch.zeros(1, dtype=torch.bool, device=self.device),
                )
            values[int(index)] = final_value.squeeze()
        return values

    def collect_rollout(self) -> RolloutCollection:
        observation, action_masks, hidden_state, episode_starts = self._require_state()
        buffer = RecurrentRolloutBuffer(
            self.config.rollout_steps,
            self.config.num_envs,
            hidden_size=self.model.hidden_size,
            recurrent_layers=self.model.recurrent_layers,
            storage_device=self.storage_device,
            global_shape=_shape3(observation["global"].shape[1:], name="global observation"),
            viewport_shape=_shape3(observation["viewport"].shape[1:], name="viewport observation"),
        )
        completed: list[Mapping[str, Any]] = []
        self.model.eval()
        for _ in range(self.config.rollout_steps):
            model_observation = observation_to(observation, self.device)
            with torch.no_grad():
                policy = self.model.get_action_and_value(
                    model_observation,
                    hidden_state,
                    episode_starts,
                    action_masks,
                )
            next_observation_raw, rewards_np, terminated_np, truncated_np, infos = (
                self.vector_env.step(policy.action.cpu().numpy())
            )
            rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
            terminated = torch.as_tensor(terminated_np, dtype=torch.bool, device=self.device)
            truncated = torch.as_tensor(truncated_np, dtype=torch.bool, device=self.device)
            truncation_values = self._truncation_values(
                infos, np.asarray(truncated_np, dtype=np.bool_), policy.hidden_state
            )
            buffer.add(
                observation,
                hidden_state,
                episode_starts,
                action_masks,
                policy.action,
                policy.log_prob,
                policy.value,
                rewards,
                terminated,
                truncated,
                truncation_values=truncation_values,
            )
            done_np = np.logical_or(terminated_np, truncated_np)
            rewards_array = np.asarray(rewards_np, dtype=np.float64)
            if rewards_array.shape != (self.config.num_envs,):
                raise ValueError(
                    "vector rewards must have shape "
                    f"[{self.config.num_envs}], got {rewards_array.shape}"
                )
            self._running_episode_returns += rewards_array
            self._running_episode_lengths += 1
            completion_step = self.global_step + self.config.num_envs
            for index in np.flatnonzero(done_np):
                environment_index = int(index)
                self._completed_episode_count += 1
                final_info = _final_info_at(infos, environment_index)
                episode = dict(final_info or {})
                # Trainer-owned values are authoritative.  WSIEnv exposes the
                # action count but Gym's final info does not supply a return.
                episode.update(
                    {
                        "episode_return": float(self._running_episode_returns[environment_index]),
                        "episode_length": int(self._running_episode_lengths[environment_index]),
                        "completed_episode_count": self._completed_episode_count,
                        "completion_global_step": completion_step,
                    }
                )
                completed.append(episode)
                self._running_episode_returns[environment_index] = 0.0
                self._running_episode_lengths[environment_index] = 0
            self.global_step += self.config.num_envs
            observation = pack_observation(next_observation_raw)
            action_masks = torch.as_tensor(
                extract_action_masks(infos, self.config.num_envs),
                dtype=torch.bool,
                device=self.device,
            )
            episode_starts = terminated | truncated
            keep = (~episode_starts).to(policy.hidden_state.dtype).view(1, self.config.num_envs, 1)
            hidden_state = policy.hidden_state * keep

        with torch.no_grad():
            last_value, _ = self.model.get_value(
                observation_to(observation, self.device),
                hidden_state,
                episode_starts,
            )
        buffer.finish(
            last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        self._observation = observation
        self._action_masks = action_masks
        self._hidden_state = hidden_state
        self._episode_starts = episode_starts
        collection = RolloutCollection(buffer=buffer, completed_episodes=tuple(completed))
        self._record_rollout_telemetry(collection.completed_episodes)
        return collection

    def update(self, rollout: RecurrentRolloutBuffer) -> PPOUpdateMetrics:
        metrics = ppo_update(self.model, self.optimizer, rollout, self.config)
        self.update_number += 1
        if self.writer is not None:
            for name, value in metrics.to_dict().items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.writer.add_scalar(f"ppo/{name}", value, self.global_step)
            self.writer.add_scalar(
                "training/learning_rate",
                float(self.optimizer.param_groups[0]["lr"]),
                self.global_step,
            )
        return metrics

    def resume(self, checkpoint: str | Path, *, restore_rng: bool = True) -> None:
        """Restore model/optimizer counters and restart vector episodes cleanly.

        OpenSlide/vector-worker state is intentionally never serialized.  The
        next collection starts newly reset episodes while retaining all learned
        parameters, optimizer moments, counters, and process RNG state.
        """

        loaded = load_ppo_checkpoint(
            checkpoint,
            self.model,
            self.optimizer,
            map_location=self.device,
            restore_rng=restore_rng,
            expected_config=self.config.to_dict(),
            compatibility_fields=RESUME_CONFIG_FIELDS,
            expected_dependencies=self.dependencies,
        )
        self.global_step = loaded.global_step
        self.update_number = loaded.update
        completed_episode_count = loaded.extra.get("completed_episode_count", 0)
        if (
            isinstance(completed_episode_count, bool)
            or not isinstance(completed_episode_count, (int, np.integer))
            or int(completed_episode_count) < 0
        ):
            raise ValueError(
                "PPO checkpoint has invalid non-negative completed_episode_count="
                f"{completed_episode_count!r}"
            )
        self._completed_episode_count = int(completed_episode_count)
        self._running_episode_returns.fill(0.0)
        self._running_episode_lengths.fill(0)
        self._observation = None
        self._action_masks = None
        self._hidden_state = None
        self._episode_starts = None

    def resume_latest_compatible(self, checkpoint_dir: str | Path) -> Path | None:
        """Resume the newest checkpoint matching configuration and fingerprints."""

        candidates = checkpoint_candidates(checkpoint_dir)
        if not candidates:
            return None
        failures: list[str] = []
        for candidate in candidates:
            try:
                self.resume(candidate)
                return candidate
            except (ValueError, RuntimeError, KeyError) as exc:
                failures.append(f"{candidate.name}: {exc}")
        raise ValueError(
            "No compatible PPO checkpoint was found. Inspected: " + "; ".join(failures)
        )

    def train(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        resume_latest: bool = True,
        update_callback: Callable[[PPOTrainer, PPOUpdateMetrics], None] | None = None,
        validation_callback: Callable[[PPOTrainer], None] | None = None,
    ) -> TrainingSummary:
        """Run configured PPO updates when explicitly invoked by the caller."""

        if checkpoint_dir is not None and resume_latest and self.global_step == 0:
            self.resume_latest_compatible(checkpoint_dir)
        remaining_timesteps = max(0, self.config.total_timesteps - self.global_step)
        updates = math.ceil(remaining_timesteps / self.config.batch_size)
        last_metrics: PPOUpdateMetrics | None = None
        next_checkpoint = (
            self.global_step // self.config.checkpoint_interval + 1
        ) * self.config.checkpoint_interval
        next_validation = (
            self.global_step // self.config.validation_interval + 1
        ) * self.config.validation_interval
        for _ in range(updates):
            if self.config.anneal_learning_rate:
                learning_rate = annealed_learning_rate(
                    self.config.learning_rate,
                    self.global_step,
                    self.config.total_timesteps,
                )
                for parameter_group in self.optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
            collection = self.collect_rollout()
            last_metrics = self.update(collection.buffer)
            if update_callback is not None:
                update_callback(self, last_metrics)
            if validation_callback is not None and self.global_step >= next_validation:
                validation_callback(self)
                while next_validation <= self.global_step:
                    next_validation += self.config.validation_interval
            if checkpoint_dir is not None and self.global_step >= next_checkpoint:
                directory = Path(checkpoint_dir)
                save_ppo_checkpoint(
                    directory / f"checkpoint_{self.global_step:012d}.pt",
                    self.model,
                    self.optimizer,
                    self.config,
                    global_step=self.global_step,
                    update=self.update_number,
                    dependencies=self.dependencies,
                    extra={"completed_episode_count": self._completed_episode_count},
                )
                while next_checkpoint <= self.global_step:
                    next_checkpoint += self.config.checkpoint_interval
        if checkpoint_dir is not None:
            directory = Path(checkpoint_dir)
            save_ppo_checkpoint(
                directory / f"checkpoint_{self.global_step:012d}.pt",
                self.model,
                self.optimizer,
                self.config,
                global_step=self.global_step,
                update=self.update_number,
                dependencies=self.dependencies,
                extra={
                    "final": True,
                    "completed_episode_count": self._completed_episode_count,
                },
            )
        return TrainingSummary(
            global_step=self.global_step,
            updates=self.update_number,
            last_metrics=last_metrics,
            completed_episodes=self._completed_episode_count,
        )

    def close(self) -> None:
        self.vector_env.close()
        if self.writer is not None and hasattr(self.writer, "flush"):
            self.writer.flush()


__all__ = [
    "RESUME_CONFIG_FIELDS",
    "EnvFactory",
    "PPOTrainer",
    "RolloutCollection",
    "TrainingSummary",
    "extract_action_masks",
    "extract_final_observation",
    "make_vector_env",
    "resolve_device",
    "seed_everything",
]
