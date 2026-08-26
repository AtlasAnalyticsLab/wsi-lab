from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rl4xs_wsi.rl.evaluation import (
    EpisodeEvaluation,
    aggregate_evaluation,
    evaluate_policy,
)


def _episode(label: int, diagnosis: int, score: float) -> EpisodeEvaluation:
    return EpisodeEvaluation(
        slide_id=f"slide-{label}",
        true_label=label,
        diagnosis=diagnosis,
        cancer_score=score,
        correct=label == diagnosis,
        episode_length=10 + label,
        return_=1.0 if label == diagnosis else -1.0,
        final_belief=score,
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


def test_policy_evaluation_resolves_auto_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    observation = {
        "global": np.zeros((5, 8, 8), dtype=np.float32),
        "viewport": np.zeros((3, 8, 8), dtype=np.float32),
        "scalars": np.asarray([0.0, 0.0, 0.25], dtype=np.float32),
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
