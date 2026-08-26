"""Lazy feature-matrix providers for existing embedding artifacts."""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import numpy as np


class HDF5FeatureMatrix:
    """Process-local, lazy access to one two-dimensional HDF5 feature dataset.

    The provider stores no live HDF5 handle when pickled, so a record can be
    passed to ``DataLoader`` workers safely.  Handles are opened only in the
    process performing row reads.  ``shape`` may be supplied from validation
    metadata to avoid even the initial metadata open.
    """

    def __init__(
        self,
        path: str | Path,
        key: str,
        *,
        shape: tuple[int, int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.key = key.strip("/")
        if not self.key:
            raise ValueError("HDF5 dataset key cannot be empty")
        self._shape: tuple[int, int] | None = (
            (int(shape[0]), int(shape[1])) if shape is not None else None
        )
        if self._shape is not None and len(self._shape) != 2:
            raise ValueError(f"HDF5 feature shape must be [N,D], got {self._shape}")
        self._process_id: int | None = None
        self._file: Any | None = None

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape is None:
            dataset = self._dataset()
            discovered = tuple(int(value) for value in dataset.shape)
            if len(discovered) != 2:
                raise ValueError(
                    f"HDF5 feature dataset {self.path}:{self.key} must be [N,D], got {discovered}"
                )
            self._shape = (discovered[0], discovered[1])
        return self._shape

    def _dataset(self) -> Any:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Reading HDF5 embeddings requires h5py; install the HDF5 package extra"
            ) from exc
        process_id = os.getpid()
        if self._file is None or self._process_id != process_id:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._process_id = process_id
        if self.key not in self._file:
            raise KeyError(f"HDF5 dataset {self.key!r} does not exist in {self.path}")
        return self._file[self.key]

    def __getitem__(self, index: Any) -> np.ndarray:
        dataset = self._dataset()
        if isinstance(index, np.ndarray) and index.ndim == 1:
            # h5py requires increasing fancy indices.  Bag order is deliberately
            # randomized after materialization, so preserving this input order is
            # unnecessary and sorting makes all h5py versions work consistently.
            index = np.sort(index.astype(np.int64, copy=False))
        return np.asarray(dataset[index])

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._process_id = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_file"] = None
        state["_process_id"] = None
        return state

    def __del__(
        self,
    ) -> None:  # pragma: no cover - interpreter shutdown is nondeterministic
        with suppress(Exception):
            self.close()


MMapMode = Literal["r+", "r", "w+", "c"]


class NumpyFeatureMatrix:
    """Memory-mapped access to a two-dimensional ``.npy`` feature array."""

    def __init__(self, path: str | Path, *, mmap_mode: MMapMode = "r") -> None:
        self.path = Path(path)
        self.mmap_mode = mmap_mode
        self._array: np.ndarray | None = None
        discovered = np.load(self.path, mmap_mode=self.mmap_mode, allow_pickle=False)
        if discovered.ndim != 2:
            raise ValueError(
                f"NumPy feature array {self.path} must be [N,D], got {discovered.shape}"
            )
        self._shape: tuple[int, int] = (
            int(discovered.shape[0]),
            int(discovered.shape[1]),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def _get_array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.path, mmap_mode=self.mmap_mode, allow_pickle=False)
        return self._array

    def __getitem__(self, index: Any) -> np.ndarray:
        return np.asarray(self._get_array()[index])

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_array"] = None
        return state


__all__ = ["HDF5FeatureMatrix", "NumpyFeatureMatrix"]
