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
    evidence_added: bool | None = None,
    evidence_count_before: int | None = None,
) -> float:
    """Return the navigation reward.

    Supplying ``evidence_added`` and ``evidence_count_before`` opts into the
    evidence-aware belief-shaping schedule used by :class:`WSIEnv`: the first
    valid embedding updates belief but is not shaped, and shaping starts with
    the second newly added embedding. Omitting both arguments retains the
    legacy helper behavior for external callers.
    """

    if reward_type == "default":
        return -step_cost
    if reward_type != "belief":
        raise ValueError(f"Unknown reward type: {reward_type}")
    if (evidence_added is None) != (evidence_count_before is None):
        raise ValueError("evidence_added and evidence_count_before must be provided together")
    if evidence_count_before is not None:
        if evidence_count_before < 0:
            raise ValueError("evidence_count_before cannot be negative")
        if not evidence_added or evidence_count_before == 0:
            return -step_cost
    truth_before = belief_before if label else 1.0 - belief_before
    truth_after = belief_after if label else 1.0 - belief_after
    return -step_cost + beta * (truth_after - truth_before)
