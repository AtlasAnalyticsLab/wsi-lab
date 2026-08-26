from __future__ import annotations

import pytest
import torch

from rl4xs_wsi.rl.distributions import MaskedCategorical, mask_logits


def test_invalid_actions_have_zero_probability_and_are_never_sampled() -> None:
    logits = torch.tensor([[1.0, 1000.0, -2.0, 1000.0]])
    mask = torch.tensor([[True, False, True, False]])
    distribution = MaskedCategorical(logits, mask)

    assert torch.equal(distribution.probs[~mask], torch.zeros(2))
    samples = torch.stack([distribution.sample() for _ in range(256)])
    assert set(samples.flatten().tolist()) <= {0, 2}
    assert torch.isfinite(distribution.entropy()).all()


def test_mask_logits_rejects_invalid_contracts() -> None:
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="at least one"):
        mask_logits(logits, torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        mask_logits(logits, torch.ones(2, 3))
    with pytest.raises(ValueError, match="match logits"):
        mask_logits(logits, torch.ones(2, 2, dtype=torch.bool))
