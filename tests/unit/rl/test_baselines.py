from __future__ import annotations

import numpy as np

from wsi_lab.baselines import (
    ConfidenceThresholdPolicy,
    FixedBudgetPolicy,
    RasterPolicy,
)


def _observation(belief: float, magnification: float = 1.0):
    return {"scalars": np.asarray([magnification, 0.5, belief], dtype=np.float32)}


def _info(valid: list[int]):
    mask = np.zeros(11, dtype=np.bool_)
    mask[valid] = True
    return {"action_mask": mask}


def test_confidence_policy_stops_only_at_threshold() -> None:
    policy = ConfidenceThresholdPolicy(0.8, navigation_policy=RasterPolicy())
    assert policy.act(_observation(0.7), _info([2, 9, 10])) == 2
    assert policy.act(_observation(0.8), _info([2, 9, 10])) == 9
    assert policy.act(_observation(0.1), _info([2, 9, 10])) == 10


def test_fixed_budget_waits_for_exact_number_of_navigation_actions() -> None:
    policy = FixedBudgetPolicy(2, navigation_policy=RasterPolicy())
    observation = _observation(0.9)
    info = _info([3, 9, 10])
    policy.reset(observation, info)
    assert policy.act(observation, info) == 3
    assert policy.act(observation, info) == 3
    assert policy.act(observation, info) == 9
