"""Exact incremental inference for a frozen gated-ABMIL belief model."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from wsi_lab.belief.model import GatedAttentionMIL, freeze_belief_model


@dataclass(frozen=True, slots=True)
class BeliefUpdate:
    probability: float
    logit: float
    evidence_count: int
    added: bool


class BeliefAccumulator:
    """Maintain the exact softmax-weighted ABMIL sufficient statistics.

    Numerator and denominator are kept relative to a running maximum attention
    logit, avoiding overflow while remaining algebraically identical to full
    bag inference.  Duplicate suppression is optional because the environment
    may already own canonical evidence identity.
    """

    def __init__(
        self,
        model: GatedAttentionMIL,
        *,
        temperature: float = 1.0,
        freeze: bool = True,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model = freeze_belief_model(model) if freeze else model
        if self.model.training:
            raise ValueError("BeliefAccumulator requires an eval-mode model")
        self.temperature = float(temperature)
        self.reset()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def evidence_count(self) -> int:
        return self._count

    @property
    def probability(self) -> float:
        if self._count == 0:
            return 0.5
        return float(torch.sigmoid(self._current_logit() / self.temperature).item())

    @property
    def logit(self) -> float:
        return 0.0 if self._count == 0 else float(self._current_logit().item())

    def reset(self) -> BeliefUpdate:
        self._maximum_attention_logit: Tensor | None = None
        self._scaled_denominator: Tensor | None = None
        self._scaled_numerator: Tensor | None = None
        self._count = 0
        self._seen: set[Hashable] = set()
        return BeliefUpdate(probability=0.5, logit=0.0, evidence_count=0, added=False)

    def _current_logit(self) -> Tensor:
        if self._scaled_numerator is None or self._scaled_denominator is None:
            return torch.zeros((), device=self.device)
        pooled = self._scaled_numerator / self._scaled_denominator
        return self.model.classify_pooled(pooled)

    def add(
        self,
        feature: Tensor | np.ndarray,
        magnification: float,
        *,
        instance_key: Hashable | None = None,
    ) -> BeliefUpdate:
        if instance_key is not None and instance_key in self._seen:
            return BeliefUpdate(
                probability=self.probability,
                logit=self.logit,
                evidence_count=self._count,
                added=False,
            )
        parameter = next(self.model.parameters())
        feature = torch.as_tensor(feature, device=self.device, dtype=parameter.dtype)
        if feature.ndim != 1 or feature.shape[0] != self.model.embedding_dim:
            raise ValueError(
                f"Expected one feature [{self.model.embedding_dim}], got {tuple(feature.shape)}"
            )
        scale = torch.tensor(float(magnification), device=self.device, dtype=parameter.dtype)
        with torch.inference_mode():
            augmented = self.model.augment_instances(feature.unsqueeze(0), scale.unsqueeze(0))
            projected = self.model.project_augmented(augmented).squeeze(0)
            attention_logit = self.model.attention_logits_from_projected(
                projected.unsqueeze(0)
            ).squeeze(0)
            if self._count == 0:
                self._maximum_attention_logit = attention_logit
                self._scaled_denominator = torch.ones_like(attention_logit)
                self._scaled_numerator = projected.clone()
            else:
                assert self._maximum_attention_logit is not None
                assert self._scaled_denominator is not None
                assert self._scaled_numerator is not None
                new_maximum = torch.maximum(self._maximum_attention_logit, attention_logit)
                old_scale = torch.exp(self._maximum_attention_logit - new_maximum)
                new_scale = torch.exp(attention_logit - new_maximum)
                self._scaled_numerator = self._scaled_numerator * old_scale + projected * new_scale
                self._scaled_denominator = self._scaled_denominator * old_scale + new_scale
                self._maximum_attention_logit = new_maximum
            self._count += 1
            if instance_key is not None:
                self._seen.add(instance_key)
            current_logit = self._current_logit()
            probability = torch.sigmoid(current_logit / self.temperature)
        return BeliefUpdate(
            probability=float(probability.item()),
            logit=float(current_logit.item()),
            evidence_count=self._count,
            added=True,
        )

    def full_bag_probability(self, features: Tensor, magnifications: Tensor) -> float:
        """Reference full recomputation using the same frozen/calibrated model."""

        parameter = next(self.model.parameters())
        with torch.inference_mode():
            logits = self.model(
                torch.as_tensor(features, device=self.device, dtype=parameter.dtype),
                torch.as_tensor(magnifications, device=self.device, dtype=parameter.dtype),
            )
            if not isinstance(logits, Tensor):
                logits = logits.logits
            return float(torch.sigmoid(logits.reshape(-1)[0] / self.temperature).item())


__all__ = ["BeliefAccumulator", "BeliefUpdate"]
