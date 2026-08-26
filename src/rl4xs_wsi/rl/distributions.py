"""Categorical action masking shared by rollout collection and PPO replay."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.distributions import Categorical


def mask_logits(logits: Tensor, action_mask: Tensor) -> Tensor:
    """Return logits with invalid actions assigned effectively zero probability.

    The shape and mask checks are intentionally strict.  A state with no valid
    action indicates an environment bug and must fail before it can generate a
    NaN-filled rollout.
    """

    if not torch.is_floating_point(logits):
        raise TypeError("logits must have a floating-point dtype")
    if action_mask.dtype != torch.bool:
        raise TypeError("action_mask must have boolean dtype")
    if logits.shape != action_mask.shape:
        raise ValueError(
            "action_mask must match logits exactly; "
            f"got mask={tuple(action_mask.shape)}, logits={tuple(logits.shape)}"
        )
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty action dimension")
    if not bool(action_mask.any(dim=-1).all()):
        raise ValueError("every policy state must have at least one valid action")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("unmasked policy logits must be finite")

    # finfo.min is preferable to a hand-picked constant across float16/32/64.
    # With at least one finite valid logit its exponent underflows to exactly
    # zero, while downstream entropy calculations remain finite.
    return logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)


class MaskedCategorical(Categorical):
    """A categorical distribution retaining the exact historical action mask."""

    arg_constraints: dict[str, Any] = {}  # noqa: RUF012

    def __init__(self, logits: Tensor, action_mask: Tensor) -> None:
        self.action_mask = action_mask
        self.unmasked_logits = logits
        super().__init__(logits=mask_logits(logits, action_mask), validate_args=False)

    @property
    def mode(self) -> Tensor:
        """Return the most probable valid action."""

        return self.logits.argmax(dim=-1)

    def entropy(self) -> Tensor:
        """Compute entropy without multiplying zero probability by huge logits."""

        log_probabilities = torch.log_softmax(self.logits, dim=-1)
        terms = torch.where(
            self.action_mask,
            self.probs * log_probabilities,
            torch.zeros_like(self.probs),
        )
        return -terms.sum(dim=-1)


__all__ = ["MaskedCategorical", "mask_logits"]
