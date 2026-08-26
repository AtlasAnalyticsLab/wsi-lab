"""Gated attention multiple-instance learning for diagnostic belief.

The policy never consumes the precomputed embeddings directly.  This module is
the single place where an embedding and its belief-eligible magnification are
combined, which keeps training and environment inference representation-identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wsi_lab.constants import BELIEF_MAGNIFICATIONS


@dataclass(frozen=True, slots=True)
class ABMILModelSpec:
    """Serializable architecture description used for compatibility checks."""

    embedding_dim: int
    hidden_dim: int = 512
    attention_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if self.hidden_dim < 1 or self.attention_dim < 1:
            raise ValueError("hidden_dim and attention_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ABMILOutput:
    """Detailed ABMIL output; all tensors retain an explicit batch axis."""

    logits: Tensor
    attention: Tensor
    pooled: Tensor
    projected: Tensor
    attention_logits: Tensor


def magnification_indices(magnifications: Tensor, *, atol: float = 1e-6) -> Tensor:
    """Map belief magnifications to the fixed four protocol indices.

    Args:
        magnifications: Tensor of any shape containing 2.5, 5, 10 or 20.
        atol: Absolute tolerance for floating-point values read from metadata.

    Raises:
        ValueError: If any value is not a belief-eligible magnification.
    """

    values = torch.as_tensor(
        BELIEF_MAGNIFICATIONS,
        dtype=(magnifications.dtype if magnifications.is_floating_point() else torch.float32),
        device=magnifications.device,
    )
    candidate = magnifications.to(dtype=values.dtype).unsqueeze(-1)
    matches = torch.isclose(candidate, values, rtol=0.0, atol=atol)
    valid = matches.sum(dim=-1) == 1
    if not bool(valid.all()):
        invalid = magnifications[~valid].detach().cpu().reshape(-1).tolist()
        raise ValueError(
            f"Unsupported belief magnification(s) {invalid}; expected {BELIEF_MAGNIFICATIONS}"
        )
    return matches.to(dtype=torch.int64).argmax(dim=-1)


def magnification_one_hot(
    magnifications: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Return the exact 2.5/5/10/20 one-hot representation."""

    indices = magnification_indices(magnifications)
    result = F.one_hot(indices, num_classes=len(BELIEF_MAGNIFICATIONS))
    return result.to(dtype=dtype or torch.float32)


def append_magnification_one_hot(features: Tensor, magnifications: Tensor) -> Tensor:
    """Concatenate embeddings with their exact four-way magnification one-hot."""

    if features.ndim < 2:
        raise ValueError(f"features must have at least two dimensions, got {features.shape}")
    if tuple(features.shape[:-1]) != tuple(magnifications.shape):
        raise ValueError(
            "magnifications must match every non-feature dimension: "
            f"features={tuple(features.shape)}, magnifications={tuple(magnifications.shape)}"
        )
    scale = magnification_one_hot(magnifications, dtype=features.dtype)
    return torch.cat((features, scale), dim=-1)


class GatedAttentionMIL(nn.Module):
    """Standard gated ABMIL over mixed-magnification slide embeddings.

    ``forward`` accepts either one bag ``[N, D]`` or a padded batch
    ``[B, N, D]``.  Returned logits always have shape ``[B]``.  A boolean mask
    is required for padded batches and each bag must contain at least one valid
    instance.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = ABMILModelSpec(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )
        augmented_dim = embedding_dim + len(BELIEF_MAGNIFICATIONS)
        self.projection = nn.Sequential(
            nn.Linear(augmented_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    @property
    def embedding_dim(self) -> int:
        return self.spec.embedding_dim

    def architecture_config(self) -> dict[str, Any]:
        """Return a JSON-serializable architecture description."""

        return self.spec.to_dict()

    @classmethod
    def from_config(
        cls,
        embedding_dim: int,
        config: Any,
    ) -> GatedAttentionMIL:
        """Build from a ``BeliefConfig``-like object or a plain mapping."""

        def get(name: str, default: Any) -> Any:
            if isinstance(config, dict):
                return config.get(name, default)
            return getattr(config, name, default)

        return cls(
            embedding_dim,
            hidden_dim=int(get("hidden_dim", 512)),
            attention_dim=int(get("attention_dim", 256)),
            dropout=float(get("dropout", 0.1)),
        )

    def augment_instances(self, features: Tensor, magnifications: Tensor) -> Tensor:
        if features.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {features.shape[-1]}"
            )
        return append_magnification_one_hot(features, magnifications)

    def project_augmented(self, augmented: Tensor) -> Tensor:
        if augmented.shape[-1] != self.embedding_dim + len(BELIEF_MAGNIFICATIONS):
            raise ValueError(
                f"Expected augmented dimension {self.embedding_dim + 4}, got {augmented.shape[-1]}"
            )
        return self.projection(augmented)

    def attention_logits_from_projected(self, projected: Tensor) -> Tensor:
        gated = torch.tanh(self.attention_v(projected)) * torch.sigmoid(self.attention_u(projected))
        return self.attention_w(gated).squeeze(-1)

    def classify_pooled(self, pooled: Tensor) -> Tensor:
        return self.classifier(pooled).squeeze(-1)

    def forward_augmented(
        self,
        augmented: Tensor,
        *,
        mask: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | ABMILOutput:
        unbatched = augmented.ndim == 2
        if unbatched:
            augmented = augmented.unsqueeze(0)
        if augmented.ndim != 3:
            raise ValueError(
                f"augmented instances must be [N,D+4] or [B,N,D+4], got {tuple(augmented.shape)}"
            )

        batch_size, instances, _ = augmented.shape
        if instances == 0:
            raise ValueError("ABMIL cannot evaluate an empty bag")
        if mask is None:
            mask = torch.ones((batch_size, instances), dtype=torch.bool, device=augmented.device)
        else:
            if unbatched and mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if tuple(mask.shape) != (batch_size, instances):
                raise ValueError(
                    f"mask must have shape {(batch_size, instances)}, got {tuple(mask.shape)}"
                )
            mask = mask.to(device=augmented.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every ABMIL bag must contain at least one valid instance")

        projected = self.project_augmented(augmented)
        attention_logits = self.attention_logits_from_projected(projected)
        masked_logits = attention_logits.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(masked_logits, dim=1)
        # Explicit zeroing prevents masked values from retaining tiny probability
        # with reduced-precision softmax implementations.
        attention = attention.masked_fill(~mask, 0.0)
        attention = attention / attention.sum(dim=1, keepdim=True)
        pooled = torch.sum(attention.unsqueeze(-1) * projected, dim=1)
        logits = self.classify_pooled(pooled)

        if return_details:
            return ABMILOutput(
                logits=logits,
                attention=attention,
                pooled=pooled,
                projected=projected,
                attention_logits=attention_logits,
            )
        return logits

    def forward(
        self,
        features: Tensor,
        magnifications: Tensor,
        *,
        mask: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | ABMILOutput:
        unbatched = features.ndim == 2
        if unbatched:
            if magnifications.ndim != 1:
                raise ValueError("An unbatched bag requires one-dimensional magnifications")
        elif features.ndim != 3 or magnifications.ndim != 2:
            raise ValueError("features/magnifications must be [N,D]/[N] or [B,N,D]/[B,N]")
        augmented = self.augment_instances(features, magnifications)
        return self.forward_augmented(
            augmented,
            mask=mask,
            return_details=return_details,
        )


def freeze_belief_model(model: GatedAttentionMIL) -> GatedAttentionMIL:
    """Put ABMIL in inference mode and disable all parameter gradients."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = [
    "ABMILModelSpec",
    "ABMILOutput",
    "GatedAttentionMIL",
    "append_magnification_one_hot",
    "freeze_belief_model",
    "magnification_indices",
    "magnification_one_hot",
]
