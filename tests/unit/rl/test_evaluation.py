from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wsi_lab.belief.evaluation import compute_belief_metrics
from wsi_lab.rl.evaluation import (
    EpisodeEvaluation,
    aggregate_evaluation,
    aggregate_ppo_patches_belief_metrics,
    evaluate_policy,
)


def _episode(
    label: int,
    diagnosis: int,
    score: float,
    *,
    final_belief: float | None = None,
    return_: float | None = None,
) -> EpisodeEvaluation:
    return EpisodeEvaluation(
        slide_id=f"slide-{label}",
        true_label=label,
        diagnosis=diagnosis,
        cancer_score=score,
        correct=label == diagnosis,
        episode_length=10 + label,
        return_=(1.0 if label == diagnosis else -1.0) if return_ is None else return_,
        final_belief=score if final_belief is None else final_belief,
        unique_evidence_count=4,
        revisit_rate=0.1,
        inspected_fraction=0.2,
        magnification_histogram={"20x": 3 + label},
    )


def test_aggregate_policy_evaluation_metrics() -> None:
    metrics = aggregate_evaluation([_episode(0, 0, 0.1), _episode(1, 1, 0.9)])
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["mean_unique_evidence_count"] == 4.0
    assert metrics["magnification_usage"] == {"20x": 7}


def test_aggregate_ppo_patches_belief_matches_standalone_belief_metrics() -> None:
    episodes = [
        _episode(0, 1, 0.9, final_belief=0.1),
        _episode(0, 0, 0.2, final_belief=0.5),
        _episode(1, 1, 0.8, final_belief=0.49),
        _episode(1, 0, 0.1, final_belief=0.9),
    ]
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.5, 0.49, 0.9], dtype=np.float64)

    metrics = aggregate_ppo_patches_belief_metrics(episodes, ece_bins=4)
    standalone = compute_belief_metrics(labels, probabilities, ece_bins=4).to_dict()

    for name, expected in standalone.items():
        assert metrics[name] == pytest.approx(expected)
    assert metrics["decision_threshold"] == 0.5
    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["counterfactual_mean_return"] == pytest.approx(0.0)


def test_aggregate_ppo_patches_belief_treats_half_as_cancerous() -> None:
    metrics = aggregate_ppo_patches_belief_metrics(
        [
            _episode(0, 0, 0.1, final_belief=0.5),
            _episode(1, 0, 0.1, final_belief=0.5),
        ]
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["auroc"] == 0.5
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.0
    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 0
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0


def test_aggregate_ppo_patches_belief_normalizes_undefined_metrics() -> None:
    metrics = aggregate_ppo_patches_belief_metrics(
        [_episode(0, 0, 0.1, final_belief=0.1)]
    )

    assert metrics["auroc"] is None
    assert metrics["sensitivity"] is None
    assert metrics["specificity"] == 1.0


def test_aggregate_ppo_patches_belief_adjusts_counterfactual_return() -> None:
    metrics = aggregate_ppo_patches_belief_metrics(
        [_episode(1, 0, 0.1, final_belief=0.9, return_=-1.25)]
    )

    assert metrics["counterfactual_mean_return"] == pytest.approx(0.75)


def test_ppo_head_metrics_remain_independent_from_final_belief() -> None:
    episodes = [
        _episode(0, 0, 0.1, final_belief=0.9),
        _episode(1, 1, 0.9, final_belief=0.1),
    ]

    assert aggregate_evaluation(episodes)["accuracy"] == 1.0
    assert aggregate_ppo_patches_belief_metrics(episodes)["accuracy"] == 0.0


def test_policy_evaluation_resolves_auto_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    observation = {
        "global": np.zeros((5, 8, 8), dtype=np.float32),
        "viewport": np.zeros((3, 8, 8), dtype=np.float32),
        "scalars": np.asarray([0.0, 0.0, 0.25, 0.5, 0.5], dtype=np.float32),
    }
    mask = np.zeros(11, dtype=np.bool_)
    mask[9] = True

    class Model:
        def __init__(self) -> None:
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def initial_state(self, batch_size: int, *, device):
            return torch.zeros(1, batch_size, 1, device=device)

        def get_action_and_value(self, packed, hidden, episode_start, action_mask, **kwargs):
            del packed, episode_start, action_mask, kwargs
            logits = torch.zeros((1, 11), device=hidden.device)
            return SimpleNamespace(
                action=torch.tensor([9], device=hidden.device),
                logits=logits,
                hidden_state=hidden,
            )

    class Environment:
        def reset(self, *, seed):
            del seed
            return observation, {"action_mask": mask}

        def step(self, action):
            assert action == 9
            return (
                observation,
                1.0,
                True,
                False,
                {"label": 1, "belief": 0.25, "slide_id": "synthetic"},
            )

    model = Model()
    result = evaluate_policy(model, Environment(), episodes=1, device="auto")

    assert model.device == torch.device("cpu")
    assert len(result.episodes) == 1
