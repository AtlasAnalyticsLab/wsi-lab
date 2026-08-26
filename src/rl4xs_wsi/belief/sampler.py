"""Deterministic, scale-aware sampling of partial slide bags."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rl4xs_wsi.constants import BELIEF_MAGNIFICATIONS


@runtime_checkable
class RowAccessibleMatrix(Protocol):
    """Minimal lazy feature-matrix interface used by the sampler.

    NumPy arrays, PyTorch tensors, memory maps, and thin HDF5 dataset wrappers
    all satisfy this contract.  HDF5 files should be opened lazily inside the
    worker that services ``__getitem__``.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: Any) -> Any: ...


@dataclass(slots=True)
class SlideEmbeddingRecord:
    """One labeled slide with sparse embeddings grouped by magnification."""

    slide_id: str
    label: int
    features_by_magnification: Mapping[float, RowAccessibleMatrix]
    patient_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.slide_id = str(self.slide_id)
        if not self.slide_id:
            raise ValueError("slide_id cannot be empty")
        self.label = int(self.label)
        if self.label not in (0, 1):
            raise ValueError(f"Slide {self.slide_id!r} label must be 0 or 1")

        normalized: dict[float, RowAccessibleMatrix] = {}
        dimensions: set[int] = set()
        for raw_magnification, matrix in self.features_by_magnification.items():
            magnification = float(raw_magnification)
            if magnification not in BELIEF_MAGNIFICATIONS:
                raise ValueError(
                    f"Slide {self.slide_id!r} has unsupported belief magnification "
                    f"{magnification}; expected {BELIEF_MAGNIFICATIONS}"
                )
            shape = tuple(int(value) for value in matrix.shape)
            if len(shape) != 2:
                raise ValueError(
                    f"Slide {self.slide_id!r} {magnification}x features must be [N,D], got {shape}"
                )
            if shape[1] < 1:
                raise ValueError(
                    f"Slide {self.slide_id!r} {magnification}x has invalid feature "
                    f"dimension {shape[1]}"
                )
            dimensions.add(shape[1])
            normalized[magnification] = matrix
        if not normalized:
            raise ValueError(f"Slide {self.slide_id!r} has no belief embeddings")
        if len(dimensions) != 1:
            raise ValueError(
                f"Slide {self.slide_id!r} embedding dimensions are inconsistent: "
                f"{sorted(dimensions)}"
            )
        self.features_by_magnification = normalized

    @property
    def embedding_dim(self) -> int:
        first = next(iter(self.features_by_magnification.values()))
        return int(first.shape[1])

    @property
    def available_counts(self) -> dict[float, int]:
        return {
            magnification: int(matrix.shape[0])
            for magnification, matrix in self.features_by_magnification.items()
        }

    @property
    def total_instances(self) -> int:
        return sum(self.available_counts.values())


@dataclass(frozen=True, slots=True)
class PartialBagSamplingConfig:
    """Bag-size bands and scale-composition parameters from the specification."""

    max_bag_size: int = 500
    band_probabilities: tuple[float, float, float] = (0.5, 0.3, 0.2)
    small_max: int = 32
    medium_max: int = 128
    scale_dirichlet_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.max_bag_size < 1:
            raise ValueError("max_bag_size must be positive")
        if not 1 <= self.small_max <= self.medium_max:
            raise ValueError("bag-size band boundaries must satisfy 1 <= small <= medium")
        probabilities = np.asarray(self.band_probabilities, dtype=np.float64)
        if probabilities.shape != (3,) or np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("band_probabilities must contain three non-negative values")
        if self.scale_dirichlet_alpha <= 0:
            raise ValueError("scale_dirichlet_alpha must be positive")

    @classmethod
    def from_config(cls, config: Any) -> PartialBagSamplingConfig:
        if isinstance(config, Mapping):
            maximum = config.get("max_bag_size", 500)
        else:
            maximum = getattr(config, "max_bag_size", 500)
        return cls(max_bag_size=int(maximum))


@dataclass(frozen=True, slots=True)
class BagSelection:
    """Unique row indices selected independently within each scale."""

    slide_id: str
    label: int
    requested_size: int
    indices_by_magnification: Mapping[float, np.ndarray]

    @property
    def actual_size(self) -> int:
        return sum(int(indices.size) for indices in self.indices_by_magnification.values())


class PartialBagSampler:
    """Sample K, then scale composition, then unique instances per scale."""

    def __init__(self, config: PartialBagSamplingConfig | None = None) -> None:
        self.config = config or PartialBagSamplingConfig()

    def sample_bag_size(self, rng: np.random.Generator) -> int:
        bounds = (
            (1, min(self.config.small_max, self.config.max_bag_size)),
            (
                self.config.small_max + 1,
                min(self.config.medium_max, self.config.max_bag_size),
            ),
            (self.config.medium_max + 1, self.config.max_bag_size),
        )
        available_bands = [index for index, (low, high) in enumerate(bounds) if low <= high]
        probabilities = np.asarray(self.config.band_probabilities, dtype=np.float64)[
            available_bands
        ]
        probabilities /= probabilities.sum()
        band = available_bands[int(rng.choice(len(available_bands), p=probabilities))]
        low, high = bounds[band]
        return int(rng.integers(low, high + 1))

    def sample(
        self,
        slide: SlideEmbeddingRecord,
        rng: np.random.Generator,
        *,
        bag_size: int | None = None,
    ) -> BagSelection:
        requested = self.sample_bag_size(rng) if bag_size is None else int(bag_size)
        if requested < 1:
            raise ValueError("bag_size must be positive")
        requested = min(requested, self.config.max_bag_size)

        counts = {
            magnification: count
            for magnification, count in slide.available_counts.items()
            if count > 0
        }
        total_available = sum(counts.values())
        if total_available == 0:
            raise ValueError(f"Slide {slide.slide_id!r} has no non-empty embedding matrices")
        target = min(requested, total_available)
        magnifications = tuple(sorted(counts))

        # Drawing composition before row indices avoids the high-resolution-cell
        # count from implicitly dominating every bag.
        weights = rng.dirichlet(
            np.full(len(magnifications), self.config.scale_dirichlet_alpha, dtype=np.float64)
        )
        proposed = rng.multinomial(target, weights)
        allocations = np.asarray(
            [min(int(proposed[index]), counts[mag]) for index, mag in enumerate(magnifications)],
            dtype=np.int64,
        )
        remaining = target - int(allocations.sum())
        while remaining:
            capacity = np.asarray(
                [counts[mag] - int(allocations[index]) for index, mag in enumerate(magnifications)],
                dtype=np.int64,
            )
            candidates = np.flatnonzero(capacity > 0)
            if not candidates.size:  # Defensive: target was already capacity-capped.
                break
            candidate_weights = weights[candidates]
            candidate_weights = candidate_weights / candidate_weights.sum()
            chosen = int(rng.choice(candidates, p=candidate_weights))
            allocations[chosen] += 1
            remaining -= 1

        # Mixed-scale bags are a model contract whenever K and availability
        # permit it.  A multinomial draw can otherwise collapse to one scale.
        populated = np.flatnonzero(allocations > 0)
        if target >= 2 and len(magnifications) >= 2 and populated.size == 1:
            source = int(populated[0])
            alternatives = np.asarray(
                [
                    index
                    for index, magnification in enumerate(magnifications)
                    if index != source and counts[magnification] > 0
                ],
                dtype=np.int64,
            )
            alternative_weights = weights[alternatives]
            alternative_weights /= alternative_weights.sum()
            destination = int(rng.choice(alternatives, p=alternative_weights))
            allocations[source] -= 1
            allocations[destination] += 1

        selected: dict[float, np.ndarray] = {}
        for index, magnification in enumerate(magnifications):
            count = int(allocations[index])
            if count:
                rows = rng.choice(counts[magnification], size=count, replace=False)
                selected[magnification] = np.asarray(rows, dtype=np.int64)
        result = BagSelection(
            slide_id=slide.slide_id,
            label=slide.label,
            requested_size=requested,
            indices_by_magnification=selected,
        )
        if result.actual_size != target:
            raise RuntimeError(
                f"Internal sampling error: selected {result.actual_size} rows, expected {target}"
            )
        return result


__all__ = [
    "BagSelection",
    "PartialBagSampler",
    "PartialBagSamplingConfig",
    "RowAccessibleMatrix",
    "SlideEmbeddingRecord",
]
