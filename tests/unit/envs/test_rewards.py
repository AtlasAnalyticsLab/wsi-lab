from __future__ import annotations

import pytest

from wsi_lab.envs.rewards import diagnosis_reward, navigation_reward


def test_default_rewards_are_exact() -> None:
    assert diagnosis_reward(1, 1) == pytest.approx(0.999)
    assert diagnosis_reward(0, 1) == pytest.approx(-1.001)
    assert navigation_reward(label=1, belief_before=0.4, belief_after=0.9) == pytest.approx(-0.001)


def test_belief_shaping_uses_truth_class_probability() -> None:
    cancer = navigation_reward(
        label=1, belief_before=0.4, belief_after=0.6, reward_type="belief", beta=1
    )
    benign = navigation_reward(
        label=0, belief_before=0.4, belief_after=0.6, reward_type="belief", beta=1
    )
    assert cancer == pytest.approx(0.199)
    assert benign == pytest.approx(-0.201)


def test_first_valid_evidence_is_not_belief_shaped() -> None:
    reward = navigation_reward(
        label=1,
        belief_before=0.5,
        belief_after=0.8,
        reward_type="belief",
        beta=1,
        evidence_added=True,
        evidence_count_before=0,
    )
    assert reward == pytest.approx(-0.001)


def test_second_valid_evidence_starts_belief_shaping() -> None:
    reward = navigation_reward(
        label=1,
        belief_before=0.6,
        belief_after=0.8,
        reward_type="belief",
        beta=1,
        evidence_added=True,
        evidence_count_before=1,
    )
    assert reward == pytest.approx(0.199)


def test_non_evidence_transition_is_not_belief_shaped() -> None:
    reward = navigation_reward(
        label=1,
        belief_before=0.6,
        belief_after=0.8,
        reward_type="belief",
        beta=1,
        evidence_added=False,
        evidence_count_before=3,
    )
    assert reward == pytest.approx(-0.001)


def test_default_reward_ignores_evidence_schedule() -> None:
    reward = navigation_reward(
        label=1,
        belief_before=0.5,
        belief_after=0.8,
        reward_type="default",
        evidence_added=True,
        evidence_count_before=0,
    )
    assert reward == pytest.approx(-0.001)
