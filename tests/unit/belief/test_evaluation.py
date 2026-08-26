from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from wsi_lab.belief.evaluation import (
    BeliefPredictions,
    binary_auroc,
    compute_belief_metrics,
    evaluate_full_bags_streaming,
    evaluate_predictions,
)
from wsi_lab.belief.model import GatedAttentionMIL
from wsi_lab.belief.sampler import SlideEmbeddingRecord


def test_exact_auroc_and_ties() -> None:
    labels = np.asarray([0, 0, 1, 1])
    assert binary_auroc(labels, np.asarray([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert binary_auroc(labels, np.ones(4)) == 0.5
    assert math.isnan(binary_auroc(np.zeros(4), np.arange(4)))


def test_binary_metrics_have_expected_values() -> None:
    metrics = compute_belief_metrics(
        np.asarray([0, 1, 0, 1]), np.asarray([0.1, 0.9, 0.2, 0.8]), ece_bins=5
    )
    assert metrics.accuracy == 1.0
    assert metrics.auroc == 1.0
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
    assert metrics.brier_score == pytest.approx(0.025)


def test_evaluation_groups_requested_not_capacity_capped_size() -> None:
    predictions = BeliefPredictions(
        logits=np.asarray([-2.0, 2.0, -1.0, 1.0]),
        probabilities=np.asarray([0.1, 0.9, 0.25, 0.75]),
        labels=np.asarray([0, 1, 0, 1]),
        requested_sizes=np.asarray([4, 4, 500, 500]),
        actual_sizes=np.asarray([4, 4, 200, 200]),
        slide_ids=("a", "b", "c", "d"),
    )
    evaluation = evaluate_predictions(predictions)
    assert set(evaluation.by_requested_size) == {4, 500}
    assert evaluation.by_requested_size[500].sample_count == 2


def test_streaming_full_bag_matches_direct_model_forward() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(19)
    model = GatedAttentionMIL(3, hidden_dim=7, attention_dim=5, dropout=0.0).eval()
    slides = []
    expected = []
    for index, label in enumerate((0, 1)):
        features_by_scale = {
            scale: np.random.default_rng(index + int(scale)).normal(size=(3, 3)).astype(np.float32)
            for scale in (2.5, 5.0, 10.0, 20.0)
        }
        slide = SlideEmbeddingRecord(f"full_{index}", label, features_by_scale)
        slides.append(slide)
        features = torch.from_numpy(np.concatenate(list(features_by_scale.values())))
        scales = torch.cat([torch.full((3,), scale) for scale in features_by_scale])
        with torch.inference_mode():
            expected.append(float(torch.sigmoid(model(features, scales)[0]).item()))
    evaluation = evaluate_full_bags_streaming(model, slides, chunk_rows=2)
    assert np.allclose(evaluation.predictions.probabilities, expected, atol=2e-6, rtol=2e-6)
    assert evaluation.predictions.actual_sizes.tolist() == [12, 12]
