from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wsi_lab.belief.accumulator import BeliefAccumulator  # noqa: E402
from wsi_lab.belief.model import GatedAttentionMIL  # noqa: E402


def test_incremental_matches_full_abmil_for_every_prefix() -> None:
    torch.manual_seed(17)
    model = GatedAttentionMIL(8, hidden_dim=16, attention_dim=9, dropout=0.1).eval()
    features = torch.randn(25, 8)
    scales = torch.tensor([2.5, 5.0, 10.0, 20.0, 5.0] * 5)
    accumulator = BeliefAccumulator(model, temperature=1.7)
    assert accumulator.probability == 0.5
    for index in range(features.shape[0]):
        update = accumulator.add(features[index], float(scales[index]), instance_key=index)
        expected = accumulator.full_bag_probability(features[: index + 1], scales[: index + 1])
        assert update.probability == pytest.approx(expected, abs=2e-6, rel=2e-6)
        assert update.evidence_count == index + 1


def test_accumulator_deduplicates_explicit_instance_keys() -> None:
    model = GatedAttentionMIL(3, hidden_dim=5, attention_dim=4, dropout=0.0).eval()
    accumulator = BeliefAccumulator(model)
    feature = torch.ones(3)
    first = accumulator.add(feature, 2.5, instance_key=("cell", 2.5))
    duplicate = accumulator.add(feature * 2, 2.5, instance_key=("cell", 2.5))
    assert first.added
    assert not duplicate.added
    assert duplicate.evidence_count == 1
    assert duplicate.probability == first.probability


def test_reset_restores_empty_bag_prior() -> None:
    model = GatedAttentionMIL(2, hidden_dim=4, attention_dim=3, dropout=0.0).eval()
    accumulator = BeliefAccumulator(model)
    accumulator.add(torch.ones(2), 20.0)
    reset = accumulator.reset()
    assert reset.probability == 0.5
    assert accumulator.evidence_count == 0
