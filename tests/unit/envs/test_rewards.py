from __future__ import annotations

import pytest

from rl4xs_wsi.envs.rewards import diagnosis_reward, navigation_reward


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
