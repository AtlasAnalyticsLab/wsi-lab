"""Paired trajectory-prefix evaluation for non-learning navigation baselines.

One trajectory is collected per navigator and slide.  Confidence-stopping and
fixed-navigation-step results are then read from prefixes of that same route,
which keeps comparisons paired and prevents a stopping rule from changing the
patches available to another condition.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from wsi_lab.baselines.base import (
    DIAGNOSIS_ACTIONS,
    BaselinePolicy,
    action_mask,
    belief,
    valid_navigation,
)
from wsi_lab.baselines.random import RandomPolicy
from wsi_lab.baselines.raster import RasterPolicy
from wsi_lab.belief.evaluation import compute_belief_metrics

DEFAULT_CONFIDENCE_THRESHOLDS = (0.5, 0.9)
DEFAULT_STEP_BUDGETS = (1, 4, 8, 16, 32, 64, 128, 256, 500)
SUPPORTED_NAVIGATORS = ("random", "raster")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SweepProtocol:
    """Stopping rules evaluated along each shared navigation trajectory."""

    confidence_thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS
    step_budgets: tuple[int, ...] = DEFAULT_STEP_BUDGETS
    confidence_min_evidence: int = 32
    confidence_max_total_actions: int = 500
    random_seed: int = 1
    ece_bins: int = 15

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) for value in self.confidence_thresholds):
            raise TypeError("confidence thresholds must be real numbers, not booleans")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.step_budgets):
            raise TypeError("step budgets must be integers")
        thresholds = tuple(float(value) for value in self.confidence_thresholds)
        budgets = tuple(int(value) for value in self.step_budgets)
        object.__setattr__(self, "confidence_thresholds", thresholds)
        object.__setattr__(self, "step_budgets", budgets)
        if not thresholds or len(set(thresholds)) != len(thresholds):
            raise ValueError("confidence_thresholds must be non-empty and unique")
        if any(not np.isfinite(value) or not 0.5 <= value <= 1.0 for value in thresholds):
            raise ValueError("confidence thresholds must lie in [0.5, 1]")
        if not budgets or len(set(budgets)) != len(budgets) or any(value <= 0 for value in budgets):
            raise ValueError("step_budgets must contain unique positive integers")
        integer_fields = {
            "confidence_min_evidence": self.confidence_min_evidence,
            "confidence_max_total_actions": self.confidence_max_total_actions,
            "random_seed": self.random_seed,
            "ece_bins": self.ece_bins,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields.values()):
            raise TypeError("sweep protocol counts and seeds must be integers")
        if self.confidence_min_evidence < 0:
            raise ValueError("confidence_min_evidence must be non-negative")
        if self.confidence_max_total_actions < 1:
            raise ValueError("confidence_max_total_actions must be positive")
        if self.ece_bins < 2:
            raise ValueError("ece_bins must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["confidence_thresholds"] = list(self.confidence_thresholds)
        value["step_budgets"] = list(self.step_budgets)
        value["confidence_diagnosis_rule"] = "p>=threshold or p<=1-threshold"
        value["terminal_diagnosis_rule"] = "p>=0.5"
        value["fixed_budget_unit"] = "navigation_actions"
        return value


@dataclass(frozen=True, slots=True)
class _State:
    slide_id: str
    true_label: int
    probability: float
    evidence_count: int
    navigation_actions: int
    revisit_rate: float | None
    inspected_fraction: float | None
    magnification_histogram: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _Endpoint:
    state: _State
    stop_reason: str
    attained: bool
    target_type: str
    target_value: float | int

    def to_dict(self) -> dict[str, Any]:
        prediction = int(self.state.probability >= 0.5)
        return {
            "slide_id": self.state.slide_id,
            "true_label": self.state.true_label,
            "final_belief": self.state.probability,
            "prediction": prediction,
            "correct": prediction == self.state.true_label,
            "unique_evidence_count": self.state.evidence_count,
            "navigation_actions": self.state.navigation_actions,
            # Every prefix is evaluated as an operational episode ending in
            # one explicit belief-based diagnosis action.
            "total_actions": self.state.navigation_actions + 1,
            "episode_length": self.state.navigation_actions + 1,
            "stop_reason": self.stop_reason,
            "attained": self.attained,
            "censored": not self.attained,
            "target_type": self.target_type,
            "target_value": self.target_value,
            "revisit_rate": self.state.revisit_rate,
            "inspected_fraction": self.state.inspected_fraction,
            "magnification_histogram": dict(self.state.magnification_histogram),
        }


@dataclass(frozen=True, slots=True)
class _EpisodePrefixes:
    confidence: Mapping[float, _Endpoint]
    fixed_steps: Mapping[int, _Endpoint]


def _stable_slide_seed(base_seed: int, slide_id: str) -> int:
    payload = f"{base_seed}\0{slide_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _navigator(name: str, *, seed: int) -> BaselinePolicy:
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "raster":
        return RasterPolicy()
    raise ValueError(f"unsupported navigator {name!r}; expected one of {SUPPORTED_NAVIGATORS}")


def _optional_float(info: Mapping[str, Any], name: str) -> float | None:
    value = info.get(name)
    return None if value is None else float(value)


def _state(
    observation: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    navigation_actions: int,
) -> _State:
    try:
        slide_id = str(info["slide_id"])
        true_label = int(info["label"])
    except KeyError as exc:
        raise KeyError("sweep environments must expose slide_id and label in info") from exc
    if true_label not in (0, 1):
        raise ValueError(f"slide {slide_id!r} has non-binary label {true_label}")
    evidence_count = int(info.get("unique_evidence_count", info.get("evidence_count", -1)))
    if evidence_count < 0:
        raise KeyError("sweep environments must expose a non-negative evidence count")
    raw_histogram = info.get("magnification_histogram", {})
    if not isinstance(raw_histogram, Mapping):
        raise TypeError("magnification_histogram must be a mapping")
    probability = float(info["belief"]) if info.get("belief") is not None else belief(observation)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("belief probability must be finite and lie in [0, 1]")
    return _State(
        slide_id=slide_id,
        true_label=true_label,
        probability=probability,
        evidence_count=evidence_count,
        navigation_actions=navigation_actions,
        revisit_rate=_optional_float(info, "revisit_rate"),
        inspected_fraction=_optional_float(info, "inspected_fraction"),
        magnification_histogram={str(key): int(value) for key, value in raw_histogram.items()},
    )


def _confidence_crossed(probability: float, threshold: float) -> bool:
    return probability >= threshold or probability <= 1.0 - threshold


def _endpoint(
    state: _State,
    *,
    stop_reason: str,
    attained: bool,
    target_type: str,
    target_value: float | int,
) -> _Endpoint:
    return _Endpoint(
        state=state,
        stop_reason=stop_reason,
        attained=attained,
        target_type=target_type,
        target_value=target_value,
    )


def _collect_episode_prefixes(
    environment: Any,
    navigator_name: str,
    protocol: SweepProtocol,
    *,
    episode_index: int,
) -> _EpisodePrefixes:
    observation, info = environment.reset(seed=protocol.random_seed + episode_index)
    initial = _state(observation, info, navigation_actions=0)
    initial_mask = action_mask(info)
    if not all(bool(initial_mask[action]) for action in DIAGNOSIS_ACTIONS):
        raise ValueError(
            "sweep environments must expose both diagnosis actions from reset; "
            "construct evaluation environments with min_diagnostic_evidence=0"
        )
    navigator = _navigator(
        navigator_name,
        seed=_stable_slide_seed(protocol.random_seed, initial.slide_id),
    )
    navigator.reset(observation, info)

    confidence: dict[float, _Endpoint] = {}
    fixed_steps: dict[int, _Endpoint] = {}
    maximum_confidence_navigation = protocol.confidence_max_total_actions - 1
    navigation_actions = 0
    previous_evidence = initial.evidence_count
    route_stop_reason: str | None = None
    current = initial

    while True:
        for budget in protocol.step_budgets:
            if budget not in fixed_steps and navigation_actions == budget:
                fixed_steps[budget] = _endpoint(
                    current,
                    stop_reason="fixed_steps",
                    attained=True,
                    target_type="navigation_steps",
                    target_value=budget,
                )

        for threshold in protocol.confidence_thresholds:
            if threshold in confidence:
                continue
            eligible = current.evidence_count >= protocol.confidence_min_evidence
            if eligible and _confidence_crossed(current.probability, threshold):
                confidence[threshold] = _endpoint(
                    current,
                    stop_reason="confidence_threshold",
                    attained=True,
                    target_type="confidence_threshold",
                    target_value=threshold,
                )
            elif navigation_actions >= maximum_confidence_navigation:
                confidence[threshold] = _endpoint(
                    current,
                    stop_reason="max_actions",
                    attained=False,
                    target_type="confidence_threshold",
                    target_value=threshold,
                )

        if len(fixed_steps) == len(protocol.step_budgets) and len(confidence) == len(
            protocol.confidence_thresholds
        ):
            break

        mask = action_mask(info)
        navigation = valid_navigation(mask)
        if not navigation:
            route_stop_reason = "environment_exhausted"
            break
        action = int(navigator.act(observation, info))
        if action in DIAGNOSIS_ACTIONS:
            route_stop_reason = (
                "raster_exhausted" if navigator_name == "raster" else "environment_exhausted"
            )
            break
        if action not in navigation:
            raise RuntimeError(f"{navigator_name} navigator selected invalid action {action}")

        observation, _, terminated, truncated, info = environment.step(action)
        navigation_actions += 1
        current = _state(observation, info, navigation_actions=navigation_actions)
        if current.slide_id != initial.slide_id or current.true_label != initial.true_label:
            raise RuntimeError("environment changed slide identity within an episode")
        if current.evidence_count < previous_evidence or current.evidence_count > previous_evidence + 1:
            raise RuntimeError("unique evidence count must increase monotonically by at most one")
        previous_evidence = current.evidence_count
        if terminated or truncated:
            route_stop_reason = "environment_terminated"
            break

    if route_stop_reason is None:
        route_stop_reason = "all_targets_attained"

    for threshold in protocol.confidence_thresholds:
        if threshold not in confidence:
            confidence[threshold] = _endpoint(
                current,
                stop_reason=route_stop_reason,
                attained=False,
                target_type="confidence_threshold",
                target_value=threshold,
            )
    for budget in protocol.step_budgets:
        if budget not in fixed_steps:
            fixed_steps[budget] = _endpoint(
                current,
                stop_reason=route_stop_reason,
                attained=False,
                target_type="navigation_steps",
                target_value=budget,
            )
    return _EpisodePrefixes(confidence=confidence, fixed_steps=fixed_steps)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _aggregate(endpoints: Sequence[_Endpoint], *, ece_bins: int) -> dict[str, Any] | None:
    if not endpoints:
        return None
    labels = np.asarray([item.state.true_label for item in endpoints], dtype=np.int64)
    probabilities = np.asarray([item.state.probability for item in endpoints], dtype=np.float64)
    predictions = probabilities >= 0.5
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.sum(predictions & positives))
    true_negative = int(np.sum(~predictions & negatives))
    false_positive = int(np.sum(predictions & negatives))
    false_negative = int(np.sum(~predictions & positives))
    classification = {
        key: _finite_or_none(value)
        for key, value in compute_belief_metrics(
            labels,
            probabilities,
            ece_bins=ece_bins,
            threshold=0.5,
        ).to_dict().items()
    }
    navigation = [float(item.state.navigation_actions) for item in endpoints]
    total = [value + 1.0 for value in navigation]
    evidence = [float(item.state.evidence_count) for item in endpoints]
    revisits = [
        item.state.revisit_rate for item in endpoints if item.state.revisit_rate is not None
    ]
    inspected = [
        item.state.inspected_fraction
        for item in endpoints
        if item.state.inspected_fraction is not None
    ]
    magnifications: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    for item in endpoints:
        magnifications.update(item.state.magnification_histogram)
        stop_reasons[item.stop_reason] += 1
    count = len(endpoints)
    attained = sum(item.attained for item in endpoints)
    capped = sum(
        item.stop_reason == "max_actions" for item in endpoints
    )
    return {
        **classification,
        "decision_threshold": 0.5,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "mean_final_belief": float(np.mean(probabilities)),
        "mean_navigation_actions": _mean(navigation),
        "median_navigation_actions": _median(navigation),
        "mean_episode_length": _mean(total),
        "median_episode_length": _median(total),
        "mean_unique_evidence_count": _mean(evidence),
        "median_unique_evidence_count": _median(evidence),
        "mean_revisit_rate": _mean(revisits),
        "mean_inspected_fraction": _mean(inspected),
        "magnification_usage": dict(sorted(magnifications.items())),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "attained_count": attained,
        "attainment_rate": attained / count,
        "censored_count": count - attained,
        "censoring_rate": (count - attained) / count,
        "action_cap_count": capped,
        "action_cap_rate": capped / count,
    }


def _attainment(endpoints: Sequence[_Endpoint], *, target_name: str) -> dict[str, Any]:
    count = len(endpoints)
    attained = sum(item.attained for item in endpoints)
    reasons = Counter(item.stop_reason for item in endpoints if not item.attained)
    capped = sum(
        item.stop_reason == "max_actions" for item in endpoints
    )
    return {
        target_name: endpoints[0].target_value if endpoints else None,
        "slide_count": count,
        "attained_count": attained,
        "censored_count": count - attained,
        "attainment_rate": attained / count if count else None,
        "censoring_rate": (count - attained) / count if count else None,
        "all_attained": attained == count,
        "action_cap_count": capped,
        "action_cap_rate": capped / count if count else None,
        "censor_reasons": dict(sorted(reasons.items())),
    }


def _navigator_report(
    environment: Any,
    navigator_name: str,
    *,
    episodes: int,
    protocol: SweepProtocol,
) -> dict[str, Any]:
    collected: list[_EpisodePrefixes] = []
    for episode_index in range(episodes):
        collected.append(
            _collect_episode_prefixes(
                environment,
                navigator_name,
                protocol,
                episode_index=episode_index,
            )
        )
        completed = episode_index + 1
        if completed % 10 == 0:
            LOGGER.info("%s baseline sweep: completed %d/%d slides", navigator_name, completed, episodes)
    confidence: dict[str, Any] = {}
    for threshold in protocol.confidence_thresholds:
        endpoints = [item.confidence[threshold] for item in collected]
        confidence[f"{threshold:g}"] = {
            "threshold": threshold,
            "confidence_min_evidence": protocol.confidence_min_evidence,
            "max_total_actions": protocol.confidence_max_total_actions,
            "episodes": [item.to_dict() for item in endpoints],
            "threshold_attainment": _attainment(endpoints, target_name="threshold"),
            "metrics": _aggregate(endpoints, ece_bins=protocol.ece_bins),
        }

    fixed_steps: dict[str, Any] = {}
    for budget in protocol.step_budgets:
        endpoints = [item.fixed_steps[budget] for item in collected]
        attained = [item for item in endpoints if item.attained]
        all_attained = len(attained) == len(endpoints)
        strict_metrics = _aggregate(endpoints, ece_bins=protocol.ece_bins) if all_attained else None
        fixed_steps[str(budget)] = {
            "target_navigation_steps": budget,
            "episodes": [item.to_dict() for item in endpoints],
            "target_attainment": _attainment(endpoints, target_name="target_navigation_steps"),
            "strict_metrics": strict_metrics,
            "attained_only_metrics": _aggregate(attained, ece_bins=protocol.ece_bins),
            "fallback_all_slides_metrics": _aggregate(endpoints, ece_bins=protocol.ece_bins),
        }
    return {
        "navigator": navigator_name,
        "confidence": confidence,
        "fixed_steps": fixed_steps,
    }


def evaluate_navigation_sweep(
    environment_factory: Callable[[], Any],
    navigator_names: Sequence[str],
    episodes: int,
    protocol: SweepProtocol,
) -> dict[str, Any]:
    """Evaluate confidence and fixed-navigation-step prefixes for each navigator.

    ``environment_factory`` is called once per navigator.  The created
    environment must allow navigation through the largest requested trajectory
    prefix and expose diagnosis actions from reset; confidence floors and caps
    are enforced logically by this evaluator.  Environments are always closed.
    """

    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    names = tuple(str(name) for name in navigator_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("navigator_names must be non-empty and unique")
    unsupported = [name for name in names if name not in SUPPORTED_NAVIGATORS]
    if unsupported:
        raise ValueError(f"unsupported navigator(s): {unsupported}")

    navigators: dict[str, Any] = {}
    for name in names:
        LOGGER.info("Starting %s baseline sweep across %d slides", name, episodes)
        environment = environment_factory()
        try:
            navigators[name] = _navigator_report(
                environment,
                name,
                episodes=episodes,
                protocol=protocol,
            )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        LOGGER.info("Finished %s baseline sweep across %d slides", name, episodes)
    return {
        "schema_version": 2,
        "protocol": protocol.to_dict(),
        "navigators": navigators,
    }


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLDS",
    "DEFAULT_STEP_BUDGETS",
    "SUPPORTED_NAVIGATORS",
    "SweepProtocol",
    "evaluate_navigation_sweep",
]
