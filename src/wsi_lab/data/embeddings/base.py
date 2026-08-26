"""Normalized embedding-reader contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from wsi_lab.data.base import EmbeddingLocation
from wsi_lab.errors import DatasetValidationError


class EmbeddingFormatError(DatasetValidationError):
    """An embedding artifact does not satisfy its declared schema."""


@dataclass(frozen=True, slots=True)
class EmbeddingMetadata:
    slide_id: str
    path: Path
    encoder: str
    feature_key: str
    coordinate_key: str
    passport_key: str | None
    magnification: float
    num_instances: int
    feature_dim: int
    feature_dtype: str
    coordinate_dtype: str
    coordinate_columns: int
    patch_size: int
    patch_size_level0: int
    level0_width: int
    level0_height: int
    level0_magnification: float
    mpp: float | None
    native_levels: tuple[int, ...]
    overlap: int
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stride_level0(self) -> int:
        if self.overlap < 0 or self.overlap >= self.patch_size:
            raise EmbeddingFormatError(
                f"invalid overlap={self.overlap} for patch_size={self.patch_size} in {self.path}"
            )
        scale = self.patch_size_level0 / self.patch_size
        return round((self.patch_size - self.overlap) * scale)


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    features: NDArray[np.floating]
    coordinates: NDArray[np.integer]
    source_indices: NDArray[np.int64]
    metadata: EmbeddingMetadata

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must have shape [N,D]")
        if self.coordinates.ndim != 2:
            raise ValueError("coordinates must have shape [N,C]")
        if len(self.features) != len(self.coordinates) or len(self.features) != len(
            self.source_indices
        ):
            raise ValueError("features, coordinates, and source_indices must have equal length")


class EmbeddingReader(ABC):
    """Read a single format without retaining unsafe cross-process handles."""

    @abstractmethod
    def inspect(self, location: EmbeddingLocation) -> EmbeddingMetadata:
        pass

    @abstractmethod
    def read(
        self, location: EmbeddingLocation, indices: Sequence[int] | None = None
    ) -> EmbeddingBatch:
        pass

    @abstractmethod
    def read_coordinates(
        self, location: EmbeddingLocation, indices: Sequence[int] | None = None
    ) -> NDArray[np.integer]:
        pass

    @abstractmethod
    def read_features(
        self, location: EmbeddingLocation, indices: Sequence[int] | None = None
    ) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def iter_feature_chunks(
        self, location: EmbeddingLocation, chunk_rows: int
    ) -> Iterator[tuple[int, int, NDArray[np.floating]]]:
        pass


class EmbeddingProvider(Protocol):
    """Slide/magnification access used by preprocessing and belief sampling."""

    def metadata(self, slide_id: str, magnification: float) -> EmbeddingMetadata: ...

    def coordinates(self, slide_id: str, magnification: float) -> NDArray[np.integer]: ...

    def features(
        self, slide_id: str, magnification: float, indices: Sequence[int] | None = None
    ) -> NDArray[np.floating]: ...

    def batch(
        self, slide_id: str, magnification: float, indices: Sequence[int] | None = None
    ) -> EmbeddingBatch: ...


def normalize_indices(indices: Sequence[int] | None, length: int) -> NDArray[np.int64] | None:
    """Validate indices while preserving caller order and duplicate requests."""

    if indices is None:
        return None
    result = np.asarray(indices, dtype=np.int64)
    if result.ndim != 1:
        raise IndexError("embedding indices must be one-dimensional")
    if np.any(result < 0) or np.any(result >= length):
        bad = result[(result < 0) | (result >= length)]
        raise IndexError(f"embedding indices out of range [0,{length}): {bad[:8].tolist()}")
    return result


__all__ = [
    "EmbeddingBatch",
    "EmbeddingFormatError",
    "EmbeddingMetadata",
    "EmbeddingProvider",
    "EmbeddingReader",
    "normalize_indices",
]
