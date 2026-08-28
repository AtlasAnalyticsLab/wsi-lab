"""Evaluation utilities for recurrent PPO and common baseline policies."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wsi_lab.baselines.base import BaselinePolicy, action_mask
from wsi_lab.belief.evaluation import compute_belief_metrics
from wsi_lab.rl.networks import RecurrentActorCritic
from wsi_lab.rl.rollout import observation_to, pack_observation


@dataclass(frozen=True)
class EpisodeEvaluation:
    slide_id: str | None
    true_label: int
    diagnosis: int
    cancer_score: float
    correct: bool
    episode_length: int
    return_: float
    final_belief: float
    unique_evidence_count: int | None
    revisit_rate: float | None
    inspected_fraction: float | None
    magnification_histogram: Mapping[str, int]


@dataclass(frozen=True)
class PolicyEvaluation:
    episodes: tuple[EpisodeEvaluation, ...]
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [asdict(episode) for episode in self.episodes],
            "metrics": dict(self.metrics),
        }


def _single_batch(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {name: np.expand_dims(np.asarray(value), axis=0) for name, value in observation.items()}


def _terminal_details(info: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = info.get("episode_metrics")
    if isinstance(metrics, Mapping):
        return {**info, **metrics}
    return info


def _first(info: Mapping[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in info and info[name] is not None:
            return info[name]
    return None


def _label(info: Mapping[str, Any], diagnosis: int) -> int:
    value = _first(info, ("label", "slide_label", "true_label", "target", "y_true"))
    if value is not None:
        label = int(value)
        if label not in (0, 1):
            raise ValueError(f"terminal label must be binary, got {label}")
        return label
    correct = _first(info, ("correct", "diagnosis_correct"))
    if correct is not None:
        return diagnosis if bool(correct) else 1 - diagnosis
    raise KeyError("terminal info must expose a binary label or correctness flag")


def _belief(observation: Mapping[str, Any], info: Mapping[str, Any]) -> float:
    value = _first(info, ("belief", "final_belief", "belief_at_diagnosis"))
    if value is None:
        value = np.asarray(observation["scalars"])[2]
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("terminal belief must lie in [0,1]")
    return probability


def _episode_record(
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    action: int,
    cancer_score: float,
    episode_return: float,
    episode_length: int,
) -> EpisodeEvaluation:
    details = _terminal_details(info)
    diagnosis = 1 if action == 9 else 0
    label = _label(details, diagnosis)
    histogram = _first(
        details,
        ("magnification_histogram", "magnification_usage", "views_by_magnification"),
    )
    return EpisodeEvaluation(
        slide_id=(str(details["slide_id"]) if details.get("slide_id") is not None else None),
        true_label=label,
        diagnosis=diagnosis,
        cancer_score=float(cancer_score),
        correct=diagnosis == label,
        episode_length=int(episode_length),
        return_=float(episode_return),
        final_belief=_belief(observation, details),
        unique_evidence_count=(
            int(value)
            if (value := _first(details, ("unique_evidence_count", "evidence_count"))) is not None
            else None
        ),
        revisit_rate=(
            float(value)
            if (value := _first(details, ("revisit_rate", "revisit_frequency"))) is not None
            else None
        ),
        inspected_fraction=(
            float(value)
            if (value := _first(details, ("inspected_fraction",))) is not None
            else None
        ),
        magnification_histogram=(
            {str(key): int(count) for key, count in histogram.items()}
            if isinstance(histogram, Mapping)
            else {}
        ),
    )


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_ppo_patches_belief_metrics(
    episodes: list[EpisodeEvaluation] | tuple[EpisodeEvaluation, ...],
    *,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Score the calibrated terminal belief on patches selected by PPO.

    The trajectory and stopping decision are unchanged.  Only the terminal
    diagnosis is replaced with ``final_belief >= 0.5``.  The counterfactual
    return adjusts the executed return by the resulting change in terminal
    diagnosis reward; all navigation rewards are preserved.
    """

    if not episodes:
        raise ValueError("at least one episode is required")
    labels = np.asarray([episode.true_label for episode in episodes], dtype=np.int64)
    probabilities = np.asarray(
        [episode.final_belief for episode in episodes], dtype=np.float64
    )
    predictions = probabilities >= 0.5
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predictions & positive))
    true_negative = int(np.sum(~predictions & negative))
    false_positive = int(np.sum(predictions & negative))
    false_negative = int(np.sum(~predictions & positive))

    belief = compute_belief_metrics(
        labels,
        probabilities,
        ece_bins=ece_bins,
        threshold=0.5,
    ).to_dict()
    # The standalone belief metrics use NaN for an undefined one-class
    # quantity.  Evaluation reports use JSON null, matching PPO metrics and
    # preventing non-finite TensorBoard values.
    normalized_belief = {
        name: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for name, value in belief.items()
    }

    ppo_predictions = np.asarray([episode.diagnosis for episode in episodes], dtype=np.int64)
    ppo_correct = ppo_predictions == labels
    belief_correct = predictions == positive
    executed_returns = np.asarray([episode.return_ for episode in episodes], dtype=np.float64)
    # Diagnosis reward is +1/-1 minus the same step cost for both readouts, so
    # replacing the terminal diagnosis changes return by +2, 0, or -2.
    counterfactual_returns = executed_returns + 2.0 * (
        belief_correct.astype(np.float64) - ppo_correct.astype(np.float64)
    )
    return {
        "decision_threshold": 0.5,
        **normalized_belief,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "counterfactual_mean_return": float(np.mean(counterfactual_returns)),
    }


def aggregate_evaluation(episodes: list[EpisodeEvaluation]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one episode is required")
    labels = np.asarray([episode.true_label for episode in episodes], dtype=np.int64)
    predictions = np.asarray([episode.diagnosis for episode in episodes], dtype=np.int64)
    scores = np.asarray([episode.cancer_score for episode in episodes], dtype=np.float64)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    )
    specificity = (
        true_negative / (true_negative + false_positive) if true_negative + false_positive else None
    )
    auroc: float | None = None
    if len(np.unique(labels)) == 2:
        try:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(labels, scores))
        except ImportError:
            auroc = None
    magnifications: Counter[str] = Counter()
    for episode in episodes:
        magnifications.update(episode.magnification_histogram)
    lengths = np.asarray([episode.episode_length for episode in episodes])
    return {
        "episodes": len(episodes),
        "accuracy": float(np.mean(labels == predictions)),
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "mean_episode_length": float(np.mean(lengths)),
        "median_episode_length": float(np.median(lengths)),
        "mean_return": float(np.mean([episode.return_ for episode in episodes])),
        "mean_unique_evidence_count": _safe_mean(
            [
                float(episode.unique_evidence_count)
                for episode in episodes
                if episode.unique_evidence_count is not None
            ]
        ),
        "mean_revisit_rate": _safe_mean(
            [episode.revisit_rate for episode in episodes if episode.revisit_rate is not None]
        ),
        "mean_inspected_fraction": _safe_mean(
            [
                episode.inspected_fraction
                for episode in episodes
                if episode.inspected_fraction is not None
            ]
        ),
        "magnification_usage": dict(sorted(magnifications.items())),
    }


def evaluate_policy(
    model: RecurrentActorCritic,
    env: Any,
    *,
    episodes: int,
    device: torch.device | str = "cpu",
    seed: int = 0,
    deterministic: bool = True,
) -> PolicyEvaluation:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    # Import lazily to avoid a module cycle while still sharing the trainer's
    # CUDA availability checks and public ``auto`` device semantics.
    from wsi_lab.rl.trainer import resolve_device

    device = resolve_device(str(device))
    model.to(device)
    model.eval()
    records: list[EpisodeEvaluation] = []
    for episode_index in range(episodes):
        observation, info = env.reset(seed=seed + episode_index)
        hidden = model.initial_state(1, device=device)
        episode_start = torch.ones(1, dtype=torch.bool, device=device)
        episode_return = 0.0
        episode_length = 0
        while True:
            mask = action_mask(info)
            packed = pack_observation(_single_batch(observation))
            with torch.no_grad():
                output = model.get_action_and_value(
                    observation_to(packed, device),
                    hidden,
                    episode_start,
                    torch.as_tensor(mask[None, :], dtype=torch.bool, device=device),
                    deterministic=deterministic,
                )
            action = int(output.action.item())
            diagnosis_logits = output.logits[0, 9:11]
            cancer_score = float(torch.softmax(diagnosis_logits, dim=-1)[0])
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            hidden = output.hidden_state
            episode_start = torch.zeros(1, dtype=torch.bool, device=device)
            if terminated or truncated:
                if action not in (9, 10):
                    raise RuntimeError("evaluation episode ended without an explicit diagnosis")
                records.append(
                    _episode_record(
                        next_observation,
                        next_info,
                        action=action,
                        cancer_score=cancer_score,
                        episode_return=episode_return,
                        episode_length=episode_length,
                    )
                )
                break
            observation, info = next_observation, next_info
    return PolicyEvaluation(tuple(records), aggregate_evaluation(records))


def evaluate_baseline(
    policy: BaselinePolicy,
    env: Any,
    *,
    episodes: int,
    seed: int = 0,
) -> PolicyEvaluation:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    records: list[EpisodeEvaluation] = []
    for episode_index in range(episodes):
        observation, info = env.reset(seed=seed + episode_index)
        policy.reset(observation, info)
        episode_return = 0.0
        episode_length = 0
        while True:
            action = int(policy.act(observation, info))
            probability = float(np.asarray(observation["scalars"])[2])
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                if action not in (9, 10):
                    raise RuntimeError("baseline episode ended without an explicit diagnosis")
                records.append(
                    _episode_record(
                        next_observation,
                        next_info,
                        action=action,
                        cancer_score=probability,
                        episode_return=episode_return,
                        episode_length=episode_length,
                    )
                )
                break
            observation, info = next_observation, next_info
    return PolicyEvaluation(tuple(records), aggregate_evaluation(records))


__all__ = [
    "EpisodeEvaluation",
    "PolicyEvaluation",
    "aggregate_evaluation",
    "aggregate_ppo_patches_belief_metrics",
    "evaluate_baseline",
    "evaluate_policy",
]
