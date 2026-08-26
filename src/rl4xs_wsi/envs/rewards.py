"""Reward definitions kept independent for numerical unit testing."""

from __future__ import annotations


def diagnosis_reward(prediction: int, label: int, step_cost: float = 0.001) -> float:
    return (1.0 if int(prediction) == int(label) else -1.0) - step_cost


def navigation_reward(
    *,
    label: int,
    belief_before: float,
    belief_after: float,
    reward_type: str = "default",
    step_cost: float = 0.001,
    beta: float = 1.0,
) -> float:
    if reward_type == "default":
        return -step_cost
    if reward_type != "belief":
        raise ValueError(f"Unknown reward type: {reward_type}")
    truth_before = belief_before if label else 1.0 - belief_before
    truth_after = belief_after if label else 1.0 - belief_after
    return -step_cost + beta * (truth_after - truth_before)
