from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from wsi_lab.baselines.sweep import SweepProtocol, evaluate_navigation_sweep


@dataclass(frozen=True)
class _Route:
    slide_id: str
    label: int
    beliefs: tuple[float, ...]
    evidence: tuple[int, ...]


class _RouteEnvironment:
    def __init__(
        self,
        routes: tuple[_Route, ...],
        *,
        raster_exhaustion: bool = False,
        branching: bool = False,
    ) -> None:
        self.routes = routes
        self.raster_exhaustion = raster_exhaustion
        self.branching = branching
        self.route_index = 0
        self.state_index = 0
        self.route = routes[0]
        self.closed = False
        self.step_actions: list[tuple[str, int]] = []

    def reset(self, *, seed: int):
        del seed
        self.route = self.routes[self.route_index % len(self.routes)]
        self.route_index += 1
        self.state_index = 0
        return self._observation(), self._info()

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "scalars": np.asarray(
                [1.0, 1.0, self.route.beliefs[self.state_index], 0.5, 0.5],
                dtype=np.float32,
            )
        }

    def _info(self) -> dict[str, Any]:
        mask = np.zeros(11, dtype=np.bool_)
        terminal = self.state_index == len(self.route.beliefs) - 1
        if self.raster_exhaustion and terminal:
            # Raster has reached the end of its route even though another
            # navigation action exists.  The sweep must treat its diagnosis
            # selection as explicit route exhaustion rather than suppress it.
            mask[0] = True
        elif self.branching:
            mask[[0, 3]] = True
        else:
            mask[3] = True
        mask[[9, 10]] = True
        return {
            "slide_id": self.route.slide_id,
            "label": self.route.label,
            "unique_evidence_count": self.route.evidence[self.state_index],
            "revisit_rate": self.state_index / (self.state_index + 1),
            "inspected_fraction": (self.state_index + 1) / 100,
            "magnification_histogram": {"20": self.state_index + 1},
            "action_mask": mask,
        }

    def step(self, action: int):
        self.step_actions.append((self.route.slide_id, action))
        if self.branching:
            self.state_index += 1
            # Make the terminal score encode the random action sequence.
            values = list(self.route.beliefs)
            values[self.state_index] = 0.2 if action == 0 else 0.8
            self.route = _Route(
                self.route.slide_id,
                self.route.label,
                tuple(values),
                self.route.evidence,
            )
        else:
            self.state_index += 1
        return self._observation(), -0.001, False, False, self._info()

    def close(self) -> None:
        self.closed = True


def test_shared_prefixes_apply_confidence_and_exact_navigation_step_stops() -> None:
    routes = (
        _Route("negative", 0, (0.5, 0.3, 0.05, 0.04, 0.02), (0, 1, 2, 3, 4)),
        _Route("positive", 1, (0.5, 0.6, 0.7, 0.8, 0.95), (0, 1, 2, 3, 4)),
    )
    environments: list[_RouteEnvironment] = []

    def factory() -> _RouteEnvironment:
        environment = _RouteEnvironment(routes)
        environments.append(environment)
        return environment

    report = evaluate_navigation_sweep(
        factory,
        ("random",),
        2,
        SweepProtocol(
            confidence_thresholds=(0.5, 0.9),
            step_budgets=(1, 2, 4),
            confidence_min_evidence=2,
            confidence_max_total_actions=4,
            random_seed=7,
            ece_bins=5,
        ),
    )

    assert report["schema_version"] == 2
    random = report["navigators"]["random"]
    threshold_half = random["confidence"]["0.5"]
    assert [episode["navigation_actions"] for episode in threshold_half["episodes"]] == [2, 2]
    assert threshold_half["threshold_attainment"]["attainment_rate"] == 1.0

    threshold_high = random["confidence"]["0.9"]
    negative, positive = threshold_high["episodes"]
    assert negative["attained"] and negative["stop_reason"] == "confidence_threshold"
    assert negative["navigation_actions"] == 2
    assert not positive["attained"] and positive["stop_reason"] == "max_actions"
    assert positive["navigation_actions"] == 3
    assert positive["total_actions"] == 4
    assert threshold_high["metrics"]["action_cap_rate"] == 0.5

    exact_two = random["fixed_steps"]["2"]
    assert all(episode["navigation_actions"] == 2 for episode in exact_two["episodes"])
    assert exact_two["strict_metrics"] is not None
    assert exact_two["strict_metrics"]["accuracy"] == 1.0

    exact_four = random["fixed_steps"]["4"]
    assert exact_four["target_attainment"]["all_attained"]
    assert exact_four["strict_metrics"]["sample_count"] == 2
    assert environments[0].closed


def test_raster_diagnosis_is_route_exhaustion_and_unattained_step_is_censored() -> None:
    routes = (
        _Route("negative", 0, (0.5, 0.4, 0.3), (0, 1, 2)),
        _Route("positive", 1, (0.5, 0.6, 0.7), (0, 1, 2)),
    )
    environments: list[_RouteEnvironment] = []

    def factory() -> _RouteEnvironment:
        environment = _RouteEnvironment(routes, raster_exhaustion=True)
        environments.append(environment)
        return environment

    report = evaluate_navigation_sweep(
        factory,
        ("raster",),
        2,
        SweepProtocol(
            confidence_thresholds=(0.9,),
            step_budgets=(1, 3),
            confidence_min_evidence=1,
            confidence_max_total_actions=10,
        ),
    )["navigators"]["raster"]

    censored = report["fixed_steps"]["3"]
    assert censored["strict_metrics"] is None
    assert censored["attained_only_metrics"] is None
    assert censored["fallback_all_slides_metrics"]["sample_count"] == 2
    assert censored["target_attainment"] == {
        "target_navigation_steps": 3,
        "slide_count": 2,
        "attained_count": 0,
        "censored_count": 2,
        "attainment_rate": 0.0,
        "censoring_rate": 1.0,
        "all_attained": False,
        "action_cap_count": 0,
        "action_cap_rate": 0.0,
        "censor_reasons": {"raster_exhausted": 2},
    }
    assert all(item["stop_reason"] == "raster_exhausted" for item in censored["episodes"])
    assert all(item["unique_evidence_count"] == 2 for item in censored["episodes"])
    assert all(
        item["stop_reason"] == "raster_exhausted"
        for item in report["confidence"]["0.9"]["episodes"]
    )
    assert len(environments[0].step_actions) == 4


def test_fixed_step_budget_does_not_mean_unique_evidence_budget() -> None:
    route = _Route("sparse-evidence", 0, (0.5, 0.4, 0.3, 0.2, 0.1), (0, 0, 0, 1, 1))
    report = evaluate_navigation_sweep(
        lambda: _RouteEnvironment((route,)),
        ("random",),
        1,
        SweepProtocol(
            confidence_thresholds=(0.9,),
            step_budgets=(4,),
            confidence_min_evidence=99,
            confidence_max_total_actions=1,
        ),
    )
    block = report["navigators"]["random"]["fixed_steps"]["4"]
    episode = block["episodes"][0]
    assert episode["attained"]
    assert episode["stop_reason"] == "fixed_steps"
    assert episode["navigation_actions"] == 4
    assert episode["total_actions"] == 5
    assert episode["unique_evidence_count"] == 1
    assert block["target_attainment"]["target_navigation_steps"] == 4
    assert report["protocol"]["fixed_budget_unit"] == "navigation_actions"


def test_random_route_is_seeded_by_slide_id_not_episode_order() -> None:
    first = _Route("slide-a", 0, (0.5, 0.5, 0.5, 0.5), (0, 1, 2, 3))
    second = _Route("slide-b", 1, (0.5, 0.5, 0.5, 0.5), (0, 1, 2, 3))
    protocol = SweepProtocol(
        confidence_thresholds=(0.9,),
        step_budgets=(3,),
        confidence_min_evidence=99,
        confidence_max_total_actions=1,
        random_seed=123,
    )

    def probabilities(routes: tuple[_Route, ...]) -> dict[str, float]:
        report = evaluate_navigation_sweep(
            lambda: _RouteEnvironment(routes, branching=True),
            ("random",),
            2,
            protocol,
        )
        episodes = report["navigators"]["random"]["fixed_steps"]["3"]["episodes"]
        return {item["slide_id"]: item["final_belief"] for item in episodes}

    assert probabilities((first, second)) == probabilities((second, first))


def test_confidence_crossing_on_navigation_499_precedes_action_cap() -> None:
    beliefs = [0.5] * 501
    beliefs[499] = 0.95
    route = _Route(
        "late-crossing",
        1,
        tuple(beliefs),
        tuple(range(501)),
    )
    report = evaluate_navigation_sweep(
        lambda: _RouteEnvironment((route,)),
        ("random",),
        1,
        SweepProtocol(
            confidence_thresholds=(0.9,),
            step_budgets=(500,),
            confidence_min_evidence=32,
            confidence_max_total_actions=500,
        ),
    )
    episode = report["navigators"]["random"]["confidence"]["0.9"]["episodes"][0]
    assert episode["attained"]
    assert episode["stop_reason"] == "confidence_threshold"
    assert episode["navigation_actions"] == 499
    assert episode["total_actions"] == 500


def test_sweep_protocol_rejects_ambiguous_or_invalid_counts() -> None:
    with pytest.raises(ValueError, match="unique"):
        SweepProtocol(step_budgets=(4, 4))
    with pytest.raises(TypeError, match="integers"):
        SweepProtocol(step_budgets=(1.5,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0.5, 1\]"):
        SweepProtocol(confidence_thresholds=(0.49,))
    with pytest.raises(ValueError, match="positive"):
        SweepProtocol(confidence_max_total_actions=0)
