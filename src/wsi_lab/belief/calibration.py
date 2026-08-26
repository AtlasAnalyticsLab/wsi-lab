"""Scalar temperature calibration for validation logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wsi_lab.errors import TestSplitAccessError
from wsi_lab.utils.atomic import atomic_json_dump


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Outcome of fitting one validation-set temperature."""

    temperature: float
    candidate_temperature: float
    nll_before: float
    nll_candidate: float
    nll_after: float
    applied: bool
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemperatureScaler(nn.Module):
    """Fixed positive scalar applied as ``calibrated_logit = logit / T``."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if not float(temperature) > 0.0:
            raise ValueError("temperature must be positive")
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature.to(device=logits.device, dtype=logits.dtype)

    def probabilities(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(self(logits))


def _validated_inputs(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    logits = torch.as_tensor(logits).detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    labels = torch.as_tensor(labels).detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if logits.numel() == 0 or logits.shape != labels.shape:
        raise ValueError("Calibration logits and labels must be non-empty and equally sized")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Calibration logits contain non-finite values")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Calibration labels must be binary")
    return logits, labels


def fit_temperature(
    logits: Tensor,
    labels: Tensor,
    *,
    max_iter: int = 100,
    minimum: float = 0.05,
    maximum: float = 20.0,
    minimum_nll_improvement: float = 1e-8,
    only_if_improves: bool = True,
    partition: str = "validation",
) -> CalibrationResult:
    """Fit temperature by deterministic full-batch validation NLL.

    A candidate is always optimized and reported.  By default it is applied
    only when it improves validation NLL, making the specification's "if
    needed" behavior explicit and preventing calibration from degrading the
    selected validation logits.
    """

    normalized_partition = partition.strip().casefold().replace("-", "_")
    if normalized_partition not in {
        "val",
        "valid",
        "validation",
        "dev",
        "development",
    }:
        raise TestSplitAccessError(
            f"Temperature calibration is validation-only; received partition {partition!r}"
        )
    logits, labels = _validated_inputs(logits, labels)
    if max_iter < 1 or not 0.0 < minimum <= maximum:
        raise ValueError("Invalid temperature optimizer bounds or iteration count")

    nll_before = float(F.binary_cross_entropy_with_logits(logits, labels).item())
    lower = torch.log(torch.tensor(minimum, dtype=torch.float64))
    upper = torch.log(torch.tensor(maximum, dtype=torch.float64))
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        bounded = torch.clamp(log_temperature, min=lower, max=upper)
        temperature = torch.exp(bounded)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        candidate = float(torch.exp(torch.clamp(log_temperature, min=lower, max=upper)).item())
        candidate_nll = float(F.binary_cross_entropy_with_logits(logits / candidate, labels).item())
    improved = candidate_nll < nll_before - minimum_nll_improvement
    applied = improved or not only_if_improves
    temperature = candidate if applied else 1.0
    nll_after = candidate_nll if applied else nll_before
    return CalibrationResult(
        temperature=temperature,
        candidate_temperature=candidate,
        nll_before=nll_before,
        nll_candidate=candidate_nll,
        nll_after=nll_after,
        applied=applied,
        sample_count=int(logits.numel()),
    )


def save_calibration(path: str | Path, result: CalibrationResult) -> None:
    atomic_json_dump(path, {"schema_version": 1, **result.to_dict()})


__all__ = [
    "CalibrationResult",
    "TemperatureScaler",
    "fit_temperature",
    "save_calibration",
]
