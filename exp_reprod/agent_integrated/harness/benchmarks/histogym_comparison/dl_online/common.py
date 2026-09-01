"""Shared policy, rollout storage, and immutable protocol helpers."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from wsi_lab.rl.distributions import MaskedCategorical
from wsi_lab.rl.networks import PolicyStep, SequenceEvaluation
from wsi_lab.rl.rollout import compute_gae

ROOT = Path(__file__).resolve().parents[3]
HARNESS_ROOT = ROOT / "benchmarks/histogym_comparison"
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved_from(source: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else source.parent / candidate).resolve()


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    episode_index: int
    slide_id: str
    slide_ordinal: int


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    path: Path
    sha256: str
    value: dict[str, Any]
    manifest_path: Path
    schedule_path: Path
    belief_checkpoint: Path
    episodes: tuple[EpisodeSpec, ...]
    checkpoints: tuple[int, ...]
    interactions_per_episode: int

    @property
    def total_interactions(self) -> int:
        return len(self.episodes) * self.interactions_per_episode


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> ExperimentProtocol:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported DL-online protocol schema: {source}")
    cohort = value["cohort"]
    workload = value["workload"]
    belief = value["belief"]
    manifest_path = _resolved_from(source, str(cohort["manifest"]))
    schedule_path = _resolved_from(source, str(workload["slide_schedule"]))
    belief_checkpoint = _resolved_from(source, str(belief["checkpoint"]))
    if sha256_file(manifest_path) != str(cohort["manifest_sha256"]):
        raise ValueError("fixed cohort manifest hash drifted")
    if sha256_file(schedule_path) != str(workload["slide_schedule_sha256"]):
        raise ValueError("fixed 10k slide schedule hash drifted")
    if not belief_checkpoint.is_file():
        raise FileNotFoundError(belief_checkpoint)
    raw_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    rows = raw_schedule.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("slide schedule has no episode rows")
    episodes = tuple(
        EpisodeSpec(
            episode_index=int(row["episode_index"]),
            slide_id=str(row["slide_id"]),
            slide_ordinal=int(row["slide_ordinal"]),
        )
        for row in rows
    )
    expected_episodes = int(workload["episodes"])
    if len(episodes) != expected_episodes:
        raise ValueError(f"schedule has {len(episodes)} episodes, expected {expected_episodes}")
    if tuple(row.episode_index for row in episodes) != tuple(range(expected_episodes)):
        raise ValueError("episode schedule indices are not contiguous")
    counts: dict[str, int] = {}
    for episode in episodes:
        counts[episode.slide_id] = counts.get(episode.slide_id, 0) + 1
    if len(counts) != int(cohort["slide_count"]):
        raise ValueError("episode schedule does not reference exactly 100 slides")
    if set(counts.values()) != {int(workload["episodes_per_slide"])}:
        raise ValueError("episode schedule is not exactly balanced across slides")
    interactions = int(workload["interactions_per_episode"])
    if len(episodes) * interactions != int(workload["total_interactions"]):
        raise ValueError("protocol total interaction count is inconsistent")
    checkpoints = tuple(int(item) for item in workload["checkpoints"])
    if checkpoints[-1] != expected_episodes:
        raise ValueError("final checkpoint must equal the episode count")
    return ExperimentProtocol(
        path=source,
        sha256=sha256_file(source),
        value=value,
        manifest_path=manifest_path,
        schedule_path=schedule_path,
        belief_checkpoint=belief_checkpoint,
        episodes=episodes,
        checkpoints=checkpoints,
        interactions_per_episode=interactions,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seeded(seed: int, factory: Callable[[], nn.Module]) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return factory()


class VisualCNN(nn.Module):
    """Small shared visual topology; only the native input channel count varies."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class NativeObservationActorCritic(nn.Module):
    """One policy core with backend-native observation branches and actor size."""

    hidden_size = 256
    recurrent_layers = 1

    def __init__(self, backend: str, *, seed: int = 1) -> None:
        super().__init__()
        if backend not in {"histogym", "wsi_lab"}:
            raise ValueError(f"unknown backend: {backend}")
        self.backend = backend
        self.num_actions = 7 if backend == "histogym" else 11
        # Shared-shape components are constructed under identical independent
        # seeds so their initial tensors are bit-identical across backends.
        self.viewport_encoder = _seeded(seed + 10, lambda: VisualCNN(3))
        if backend == "wsi_lab":
            self.global_encoder = _seeded(seed + 11, lambda: VisualCNN(5))
            self.scalar_encoder = _seeded(
                seed + 12,
                lambda: nn.Sequential(
                    nn.Linear(5, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                ),
            )
            fusion_inputs = 256 + 256 + 64
        else:
            self.global_encoder = None
            self.scalar_encoder = None
            fusion_inputs = 256
        self.fusion = _seeded(
            seed + 13,
            lambda: nn.Sequential(nn.Linear(fusion_inputs, 256), nn.ReLU()),
        )
        self.recurrent = _seeded(seed + 14, lambda: nn.GRU(256, 256, num_layers=1))
        self.actor = _seeded(seed + 15, lambda: nn.Linear(256, self.num_actions))
        self.critic = _seeded(seed + 16, lambda: nn.Linear(256, 1))

    @staticmethod
    def _image(value: Tensor) -> Tensor:
        if value.dtype == torch.uint8:
            return value.float().div(255.0)
        if not torch.is_floating_point(value):
            raise TypeError("image input must be uint8 or floating point")
        return value.float()

    @staticmethod
    def _flatten_branch(value: Tensor) -> tuple[Tensor, int, int, bool]:
        if value.ndim == 5:
            time_steps, batch_size = value.shape[:2]
            return value.reshape(-1, *value.shape[2:]), time_steps, batch_size, True
        if value.ndim == 4:
            batch_size = value.shape[0]
            return value, 1, batch_size, False
        raise ValueError(f"image branch must be [B,C,H,W] or [T,B,C,H,W], got {value.shape}")

    def encode(self, observation: Mapping[str, Tensor]) -> Tensor:
        viewport_key = "image" if self.backend == "histogym" else "viewport"
        viewport, time_steps, batch_size, sequence = self._flatten_branch(observation[viewport_key])
        features = [self.viewport_encoder(self._image(viewport))]
        if self.backend == "wsi_lab":
            assert self.global_encoder is not None and self.scalar_encoder is not None
            global_image, global_time, global_batch, global_sequence = self._flatten_branch(
                observation["global"]
            )
            if (global_time, global_batch, global_sequence) != (
                time_steps,
                batch_size,
                sequence,
            ):
                raise ValueError("WSI-Lab visual branch dimensions differ")
            scalars = observation["scalars"]
            expected_scalar_shape = (time_steps, batch_size, 5) if sequence else (batch_size, 5)
            if tuple(scalars.shape) != expected_scalar_shape:
                raise ValueError(
                    f"scalars must have shape {expected_scalar_shape}, got {tuple(scalars.shape)}"
                )
            features.extend(
                [
                    self.global_encoder(self._image(global_image)),
                    self.scalar_encoder(scalars.reshape(-1, 5).float()),
                ]
            )
        fused = self.fusion(torch.cat(features, dim=-1))
        return fused.reshape(time_steps, batch_size, 256)

    def recurrent_features(
        self,
        encoded: Tensor,
        hidden_state: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        time_steps, batch_size = encoded.shape[:2]
        if tuple(hidden_state.shape) != (1, batch_size, self.hidden_size):
            raise ValueError("hidden-state shape does not match recurrent batch")
        if tuple(episode_starts.shape) != (time_steps, batch_size):
            raise ValueError("episode-start shape does not match recurrent input")
        hidden = hidden_state
        outputs: list[Tensor] = []
        for step in range(time_steps):
            keep = (~episode_starts[step]).to(hidden.dtype).view(1, batch_size, 1)
            hidden = hidden * keep
            output, hidden = self.recurrent(encoded[step : step + 1], hidden)
            outputs.append(output)
        return torch.cat(outputs, dim=0), hidden

    def initial_state(self, batch_size: int, *, device: torch.device | str) -> Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

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
        encoded = self.encode(observation)
        recurrent, final_hidden = self.recurrent_features(
            encoded, hidden_state, episode_starts.unsqueeze(0)
        )
        features = recurrent.squeeze(0)
        logits = self.actor(features)
        distribution = MaskedCategorical(logits, action_mask)
        selected = (
            distribution.mode
            if action is None and deterministic
            else distribution.sample()
            if action is None
            else action.long()
        )
        return PolicyStep(
            action=selected,
            log_prob=distribution.log_prob(selected),
            entropy=distribution.entropy(),
            value=self.critic(features).squeeze(-1),
            logits=distribution.logits,
            hidden_state=final_hidden,
        )

    def get_value(
        self,
        observation: Mapping[str, Tensor],
        hidden_state: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        encoded = self.encode(observation)
        recurrent, final_hidden = self.recurrent_features(
            encoded, hidden_state, episode_starts.unsqueeze(0)
        )
        return self.critic(recurrent.squeeze(0)).squeeze(-1), final_hidden

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
        distribution = MaskedCategorical(logits, action_mask)
        return SequenceEvaluation(
            log_prob=distribution.log_prob(actions.long()),
            entropy=distribution.entropy(),
            value=self.critic(recurrent).squeeze(-1),
            logits=distribution.logits,
            hidden_state=final_hidden,
        )

    def metadata(self) -> dict[str, Any]:
        def module_hash(module: nn.Module | None) -> str | None:
            if module is None:
                return None
            digest = hashlib.sha256()
            for name, tensor in sorted(module.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        return {
            "backend": self.backend,
            "num_actions": self.num_actions,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "initial_component_sha256": {
                "viewport_encoder": module_hash(self.viewport_encoder),
                "global_encoder": module_hash(self.global_encoder),
                "scalar_encoder": module_hash(self.scalar_encoder),
                "fusion": module_hash(self.fusion),
                "recurrent": module_hash(self.recurrent),
                "actor": module_hash(self.actor),
                "critic": module_hash(self.critic),
            },
        }


def _pack_image(value: Any, channels: int) -> Tensor:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != channels or tuple(array.shape[1:]) != (224, 224):
        raise ValueError(f"expected [{channels},224,224] image, got {array.shape}")
    if array.dtype == np.uint8:
        packed = np.ascontiguousarray(array)
    elif np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("visual observation contains non-finite values")
        packed = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        raise TypeError(f"unsupported image dtype {array.dtype}")
    return torch.from_numpy(packed)


def pack_observation(backend: str, observation: Any) -> dict[str, Tensor]:
    if backend == "histogym":
        return {"image": _pack_image(observation, 3)}
    if backend != "wsi_lab" or not isinstance(observation, Mapping):
        raise TypeError("WSI-Lab observation must be a mapping")
    scalars = np.asarray(observation["scalars"], dtype=np.float32)
    if scalars.shape != (5,) or not np.isfinite(scalars).all():
        raise ValueError("WSI-Lab scalar observation must contain five finite values")
    return {
        "global": _pack_image(observation["global"], 5),
        "viewport": _pack_image(observation["viewport"], 3),
        "scalars": torch.from_numpy(np.ascontiguousarray(scalars)),
    }


def observation_nbytes(observation: Any) -> int:
    if isinstance(observation, Mapping):
        return sum(int(np.asarray(value).nbytes) for value in observation.values())
    return int(np.asarray(observation).nbytes)


def observation_to_device(
    observation: Mapping[str, Tensor], device: torch.device | str
) -> dict[str, Tensor]:
    return {
        name: value.unsqueeze(0).to(device=device, non_blocking=False)
        for name, value in observation.items()
    }


@dataclass(frozen=True)
class SequenceBatch:
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

    def to(self, device: torch.device | str) -> SequenceBatch:
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, dict):
                values[field.name] = {key: item.to(device) for key, item in value.items()}
            else:
                values[field.name] = value.to(device)
        return SequenceBatch(**values)


class NativeRolloutBuffer:
    """Generic one-environment recurrent buffer compatible with local PPO."""

    num_envs = 1

    def __init__(
        self,
        rollout_steps: int,
        observation_shapes: Mapping[str, Sequence[int]],
        num_actions: int,
    ) -> None:
        self.rollout_steps = int(rollout_steps)
        self.num_actions = int(num_actions)
        self.observations = {
            name: torch.empty(
                self.rollout_steps,
                1,
                *tuple(int(item) for item in shape),
                dtype=torch.float32 if name == "scalars" else torch.uint8,
            )
            for name, shape in observation_shapes.items()
        }
        shape = (self.rollout_steps, 1)
        self.hidden_states = torch.empty(self.rollout_steps, 1, 1, 256)
        self.episode_starts = torch.empty(*shape, dtype=torch.bool)
        self.action_masks = torch.empty(*shape, self.num_actions, dtype=torch.bool)
        self.actions = torch.empty(*shape, dtype=torch.long)
        self.log_probs = torch.empty(*shape)
        self.values = torch.empty(*shape)
        self.rewards = torch.empty(*shape)
        self.terminated = torch.empty(*shape, dtype=torch.bool)
        self.truncated = torch.empty(*shape, dtype=torch.bool)
        self.truncation_values = torch.zeros(*shape)
        self.advantages = torch.empty(*shape)
        self.returns = torch.empty(*shape)
        self._cursor = 0
        self._finished = False

    @property
    def full(self) -> bool:
        return self._cursor == self.rollout_steps

    def reset(self) -> None:
        self._cursor = 0
        self._finished = False

    def add(
        self,
        observation: Mapping[str, Tensor],
        hidden_state: Tensor,
        episode_start: bool,
        action_mask: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: float,
        *,
        terminated: bool,
        truncated: bool,
        truncation_value: Tensor | None,
    ) -> None:
        if self.full:
            raise RuntimeError("rollout buffer is full")
        if set(observation) != set(self.observations):
            raise ValueError("rollout observation branches changed")
        step = self._cursor
        for name, tensor in observation.items():
            self.observations[name][step, 0].copy_(tensor)
        self.hidden_states[step].copy_(hidden_state.detach().cpu())
        self.episode_starts[step, 0] = bool(episode_start)
        self.action_masks[step, 0].copy_(action_mask.detach().cpu().squeeze(0))
        self.actions[step, 0] = int(action.item())
        self.log_probs[step, 0] = float(log_prob.item())
        self.values[step, 0] = float(value.item())
        self.rewards[step, 0] = float(reward)
        self.terminated[step, 0] = bool(terminated)
        self.truncated[step, 0] = bool(truncated)
        self.truncation_values[step, 0] = (
            0.0 if truncation_value is None else float(truncation_value.item())
        )
        self._cursor += 1

    def finish(self, last_value: Tensor, *, gamma: float, gae_lambda: float) -> None:
        if not self.full:
            raise RuntimeError("cannot finish an incomplete rollout")
        next_values = torch.empty_like(self.values)
        next_values[:-1].copy_(self.values[1:])
        next_values[-1, 0] = float(last_value.item())
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

    def _batch(self, starts: Sequence[int], sequence_length: int) -> SequenceBatch:
        def sequences(tensor: Tensor) -> Tensor:
            return torch.stack(
                [tensor[start : start + sequence_length, 0] for start in starts], dim=1
            )

        observations = {name: sequences(value) for name, value in self.observations.items()}
        initial_hidden = torch.stack([self.hidden_states[start, :, 0] for start in starts], dim=1)
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
        if not self._finished:
            raise RuntimeError("finish() must be called before PPO sampling")
        if self.rollout_steps % sequence_length:
            raise ValueError("sequence length must divide rollout length")
        starts = list(range(0, self.rollout_steps, sequence_length))
        order = torch.randperm(len(starts), generator=generator).tolist()
        shuffled = [starts[index] for index in order]
        for positions in np.array_split(np.arange(len(shuffled)), num_minibatches):
            if positions.size:
                yield self._batch(
                    [shuffled[int(position)] for position in positions], sequence_length
                )


def nanosecond_summary(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.int64)
    if array.size == 0:
        return {"count": 0, "sum_ns": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    return {
        "count": int(array.size),
        "sum_ns": int(array.sum(dtype=np.int64)),
        "mean_ms": float(array.mean() / 1e6),
        "median_ms": float(np.median(array) / 1e6),
        "p95_ms": float(np.percentile(array, 95) / 1e6),
    }


def action_counts(actions: np.ndarray, num_actions: int) -> dict[str, int]:
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=num_actions)
    return {str(index): int(counts[index]) for index in range(num_actions)}


def ppo_config(protocol: ExperimentProtocol):
    from wsi_lab.rl.recurrent_ppo import PPOConfig

    values = dict(protocol.value["ppo"])
    return PPOConfig(
        seed=int(protocol.value["policy"]["seed"]),
        num_envs=int(values["num_envs"]),
        rollout_steps=int(values["rollout_steps"]),
        sequence_length=int(values["sequence_length"]),
        learning_rate=float(values["learning_rate"]),
        gamma=float(values["gamma"]),
        gae_lambda=float(values["gae_lambda"]),
        clip_coef=float(values["clip_coef"]),
        update_epochs=int(values["update_epochs"]),
        entropy_coef=float(values["entropy_coef"]),
        value_coef=float(values["value_coef"]),
        max_grad_norm=float(values["max_grad_norm"]),
        num_minibatches=int(values["num_minibatches"]),
        total_timesteps=protocol.total_interactions,
        anneal_learning_rate=bool(values["anneal_learning_rate"]),
        clip_value_loss=bool(values["clip_value_loss"]),
        target_kl=values["target_kl"],
        checkpoint_interval=protocol.total_interactions,
        validation_interval=protocol.total_interactions,
        vector_backend="sync",
        device=str(protocol.value["policy"]["device"]),
    )


__all__ = [
    "DEFAULT_PROTOCOL",
    "EpisodeSpec",
    "ExperimentProtocol",
    "NativeObservationActorCritic",
    "NativeRolloutBuffer",
    "action_counts",
    "canonical_sha256",
    "load_protocol",
    "nanosecond_summary",
    "observation_nbytes",
    "observation_to_device",
    "pack_observation",
    "ppo_config",
    "seed_everything",
    "sha256_file",
]
