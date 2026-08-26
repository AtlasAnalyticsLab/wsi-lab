"""AtlasPatch HDF5 reader for the discovered CAMELYON16 artifacts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from rl4xs_wsi.data.base import DatasetAdapter, EmbeddingLocation
from rl4xs_wsi.data.embeddings.base import (
    EmbeddingBatch,
    EmbeddingFormatError,
    EmbeddingMetadata,
    EmbeddingReader,
    normalize_indices,
)


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _ordered_hdf5_selection(
    dataset: h5py.Dataset, requested: NDArray[np.int64] | None
) -> np.ndarray:
    """Read arbitrary row order despite h5py's increasing-index requirement."""

    if requested is None:
        return np.asarray(dataset[:])
    if len(requested) == 0:
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    unique, inverse = np.unique(requested, return_inverse=True)
    selected = np.asarray(dataset[unique])
    return selected[inverse]


class AtlasPatchHDF5Reader(EmbeddingReader):
    """Validate and read AtlasPatch's per-slide HDF5 schema."""

    def inspect(self, location: EmbeddingLocation) -> EmbeddingMetadata:
        path = Path(location.path)
        if not path.is_file():
            raise EmbeddingFormatError(f"embedding file does not exist: {path}")
        try:
            with h5py.File(path, "r") as handle:
                coordinates = self._dataset(handle, location.coordinate_key, path)
                features = self._dataset(handle, location.feature_key, path)
                if coordinates.ndim != 2 or coordinates.shape[1] < 2:
                    raise EmbeddingFormatError(
                        f"coordinate dataset must have shape [N,C>=2], got {coordinates.shape} in {path}"
                    )
                if features.ndim != 2:
                    raise EmbeddingFormatError(
                        f"feature dataset must have shape [N,D], got {features.shape} in {path}"
                    )
                if coordinates.shape[0] != features.shape[0]:
                    raise EmbeddingFormatError(
                        f"row mismatch in {path}: {location.coordinate_key}={coordinates.shape[0]}, "
                        f"{location.feature_key}={features.shape[0]}"
                    )
                passport_key = location.passport_key
                if passport_key:
                    passports = self._dataset(handle, passport_key, path)
                    if passports.ndim != 1 or passports.shape[0] != coordinates.shape[0]:
                        raise EmbeddingFormatError(
                            f"passport dataset must have shape [N], got {passports.shape} in {path}"
                        )
                attrs = {str(key): _plain(value) for key, value in handle.attrs.items()}
                metadata = self._metadata(location, path, coordinates, features, attrs)
                declared = attrs.get("num_patches")
                if declared is not None and int(declared) != metadata.num_instances:
                    raise EmbeddingFormatError(
                        f"num_patches={declared} but datasets contain {metadata.num_instances} rows in {path}"
                    )
                return metadata
        except OSError as exc:
            raise EmbeddingFormatError(f"cannot open HDF5 embedding file {path}: {exc}") from exc

    def read(
        self, location: EmbeddingLocation, indices: Sequence[int] | None = None
    ) -> EmbeddingBatch:
        metadata = self.inspect(location)
        requested = normalize_indices(indices, metadata.num_instances)
        try:
            with h5py.File(location.path, "r") as handle:
                coordinates = _ordered_hdf5_selection(handle[location.coordinate_key], requested)
                features = _ordered_hdf5_selection(handle[location.feature_key], requested)
        except OSError as exc:
            raise EmbeddingFormatError(
                f"cannot read HDF5 embedding file {location.path}: {exc}"
            ) from exc
        source = (
            np.arange(metadata.num_instances, dtype=np.int64)
            if requested is None
            else requested.copy()
        )
        return EmbeddingBatch(
            features=features, coordinates=coordinates, source_indices=source, metadata=metadata
        )

    def read_coordinates(
        self, location: EmbeddingLocation, indices: Sequence[int] | None = None
    ) -> NDArray[np.integer]:
        with h5py.File(location.path, "r") as handle:
            dataset = self._dataset(handle, location.coordinate_key, Path(location.path))
            requested = normalize_indices(indices, int(dataset.shape[0]))
            return _ordered_hdf5_selection(dataset, requested)

    def read_features(
        self, location: EmbeddingLocation, indices: Sequence[int] | None = None
    ) -> NDArray[np.floating]:
        with h5py.File(location.path, "r") as handle:
            dataset = self._dataset(handle, location.feature_key, Path(location.path))
            requested = normalize_indices(indices, int(dataset.shape[0]))
            return _ordered_hdf5_selection(dataset, requested)

    def read_passports(
        self, location: EmbeddingLocation, indices: Sequence[int] | None = None
    ) -> np.ndarray | None:
        if not location.passport_key:
            return None
        with h5py.File(location.path, "r") as handle:
            dataset = self._dataset(handle, location.passport_key, Path(location.path))
            requested = normalize_indices(indices, int(dataset.shape[0]))
            return _ordered_hdf5_selection(dataset, requested)

    def iter_feature_chunks(
        self, location: EmbeddingLocation, chunk_rows: int
    ) -> Iterator[tuple[int, int, NDArray[np.floating]]]:
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        metadata = self.inspect(location)
        with h5py.File(location.path, "r") as handle:
            dataset = handle[location.feature_key]
            for start in range(0, metadata.num_instances, chunk_rows):
                end = min(metadata.num_instances, start + chunk_rows)
                yield start, end, np.asarray(dataset[start:end])

    @staticmethod
    def _dataset(handle: h5py.File, key: str, path: Path) -> h5py.Dataset:
        normalized = key.strip("/")
        if normalized not in handle:
            raise EmbeddingFormatError(f"missing HDF5 dataset {key!r} in {path}")
        value = handle[normalized]
        if not isinstance(value, h5py.Dataset):
            raise EmbeddingFormatError(f"HDF5 key {key!r} is not a dataset in {path}")
        return value

    @staticmethod
    def _metadata(
        location: EmbeddingLocation,
        path: Path,
        coordinates: h5py.Dataset,
        features: h5py.Dataset,
        attrs: dict[str, Any],
    ) -> EmbeddingMetadata:
        required = (
            "patch_size",
            "patch_size_level0",
            "level0_width",
            "level0_height",
            "level0_magnification",
        )
        missing = [key for key in required if key not in attrs]
        if missing:
            raise EmbeddingFormatError(f"missing required HDF5 attributes {missing} in {path}")
        observed_mag = attrs.get("target_magnification", attrs.get("magnification"))
        if observed_mag is None:
            raise EmbeddingFormatError(f"missing target_magnification attribute in {path}")
        if not np.isclose(float(observed_mag), location.magnification, rtol=0.0, atol=1e-6):
            raise EmbeddingFormatError(
                f"magnification mismatch for {path}: configured={location.magnification:g}, "
                f"artifact={float(observed_mag):g}"
            )
        native_levels: tuple[int, ...] = ()
        if coordinates.shape[0] and coordinates.shape[1] >= 5:
            # Only the fifth column is read; feature tensors remain lazy.
            native_levels = tuple(int(value) for value in np.unique(coordinates[:, 4]))
        return EmbeddingMetadata(
            slide_id=location.slide_id,
            path=path,
            encoder=location.encoder,
            feature_key=location.feature_key,
            coordinate_key=location.coordinate_key,
            passport_key=location.passport_key,
            magnification=float(observed_mag),
            num_instances=int(features.shape[0]),
            feature_dim=int(features.shape[1]),
            feature_dtype=str(features.dtype),
            coordinate_dtype=str(coordinates.dtype),
            coordinate_columns=int(coordinates.shape[1]),
            patch_size=int(attrs["patch_size"]),
            patch_size_level0=int(attrs["patch_size_level0"]),
            level0_width=int(attrs["level0_width"]),
            level0_height=int(attrs["level0_height"]),
            level0_magnification=float(attrs["level0_magnification"]),
            mpp=float(attrs["mpp"]) if attrs.get("mpp") is not None else None,
            native_levels=native_levels,
            overlap=int(attrs.get("overlap", 0)),
            attributes=attrs,
        )


class HDF5EmbeddingProvider:
    """Resolve adapter locations and delegate read operations."""

    def __init__(self, adapter: DatasetAdapter, reader: AtlasPatchHDF5Reader | None = None) -> None:
        self.adapter = adapter
        self.reader = reader or AtlasPatchHDF5Reader()
        self._metadata: dict[tuple[str, float], EmbeddingMetadata] = {}

    def location(self, slide_id: str, magnification: float) -> EmbeddingLocation:
        return self.adapter.embedding_location(slide_id, magnification)

    def metadata(self, slide_id: str, magnification: float) -> EmbeddingMetadata:
        key = (slide_id, float(magnification))
        if key not in self._metadata:
            self._metadata[key] = self.reader.inspect(self.location(*key))
        return self._metadata[key]

    def coordinates(self, slide_id: str, magnification: float) -> NDArray[np.integer]:
        return self.reader.read_coordinates(self.location(slide_id, magnification))

    def features(
        self, slide_id: str, magnification: float, indices: Sequence[int] | None = None
    ) -> NDArray[np.floating]:
        return self.reader.read_features(self.location(slide_id, magnification), indices)

    def batch(
        self, slide_id: str, magnification: float, indices: Sequence[int] | None = None
    ) -> EmbeddingBatch:
        return self.reader.read(self.location(slide_id, magnification), indices)


__all__ = ["AtlasPatchHDF5Reader", "HDF5EmbeddingProvider"]
