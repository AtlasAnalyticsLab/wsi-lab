"""Benchmark-only EpisodeSource adapter that blocks online feature reads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class FixedEmbeddingEpisodeSource:
    """Delegate native WSI work while replacing every embedding with one RAM value.

    The wrapped ``ProcessedEpisodeSource`` still supplies the native slide geometry,
    mmap artifacts, OpenSlide viewports, and lifecycle.  Its ``read_embedding``
    method is deliberately unreachable through this adapter, guaranteeing that the
    online benchmark never reads a UNI-v2 feature row.
    """

    def __init__(self, source: Any, *, embedding_dimension: int = 1) -> None:
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        self._source = source
        self._dummy: np.ndarray = np.zeros((embedding_dimension,), dtype=np.float32)
        self._dummy.setflags(write=False)
        self.dummy_embedding_returns = 0
        self.underlying_embedding_reads = 0

    @property
    def wrapped_source(self) -> Any:
        return self._source

    def slide_ids(self, split: str) -> Sequence[str]:
        return self._source.slide_ids(split)

    def load_slide(self, slide_id: str) -> Any:
        return self._source.load_slide(slide_id)

    def read_viewport(self, slide_id: str, cell: Any, output_size: int) -> np.ndarray:
        return self._source.read_viewport(slide_id, cell, output_size)

    def read_embedding(self, slide_id: str, cell: Any) -> np.ndarray:
        del slide_id, cell
        self.dummy_embedding_returns += 1
        # ConstantBelief ignores the feature. Returning the same read-only array
        # avoids random-number work and guarantees no HDF5 feature access.
        return self._dummy

    def close(self) -> None:
        close = getattr(self._source, "close", None)
        if callable(close):
            close()


__all__ = ["FixedEmbeddingEpisodeSource"]
