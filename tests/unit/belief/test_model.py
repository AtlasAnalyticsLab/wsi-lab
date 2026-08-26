from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wsi_lab.belief.model import (  # noqa: E402
    GatedAttentionMIL,
    append_magnification_one_hot,
    magnification_one_hot,
)


def test_magnification_one_hot_is_exact_protocol_order() -> None:
    scales = torch.tensor([2.5, 5.0, 10.0, 20.0])
    assert torch.equal(magnification_one_hot(scales), torch.eye(4))
    features = torch.zeros(4, 3)
    augmented = append_magnification_one_hot(features, scales)
    assert augmented.shape == (4, 7)
    assert torch.equal(augmented[:, -4:], torch.eye(4))


def test_magnification_one_hot_rejects_navigation_only_scale() -> None:
    with pytest.raises(ValueError, match="Unsupported belief magnification"):
        magnification_one_hot(torch.tensor([1.25]))


def test_abmil_variable_bags_mask_padding_and_backward() -> None:
    torch.manual_seed(3)
    model = GatedAttentionMIL(embedding_dim=6, hidden_dim=12, attention_dim=7, dropout=0.0)
    features = torch.randn(2, 5, 6, requires_grad=True)
    scales = torch.tensor([[2.5, 5.0, 10.0, 20.0, 2.5], [5.0, 10.0, 2.5, 2.5, 2.5]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    details = model(features, scales, mask=mask, return_details=True)
    assert details.logits.shape == (2,)
    assert details.attention.shape == (2, 5)
    assert torch.equal(details.attention[1, 2:], torch.zeros(3))
    assert torch.allclose(details.attention.sum(dim=1), torch.ones(2))
    details.logits.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_padding_values_do_not_change_output() -> None:
    torch.manual_seed(7)
    model = GatedAttentionMIL(embedding_dim=4, hidden_dim=8, attention_dim=5, dropout=0.0).eval()
    valid = torch.randn(3, 4)
    scales = torch.tensor([2.5, 5.0, 10.0])
    expected = model(valid, scales)
    padded = torch.cat((valid, torch.randn(4, 4) * 100), dim=0).unsqueeze(0)
    padded_scales = torch.tensor([[2.5, 5.0, 10.0, 20.0, 20.0, 20.0, 20.0]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    actual = model(padded, padded_scales, mask=mask)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_empty_bag_is_rejected() -> None:
    model = GatedAttentionMIL(embedding_dim=2, hidden_dim=4, attention_dim=3)
    with pytest.raises(ValueError, match="empty bag"):
        model(torch.empty(0, 2), torch.empty(0))
