"""Embedding format adapters."""

from rl4xs_wsi.data.embeddings.base import (
    EmbeddingBatch,
    EmbeddingFormatError,
    EmbeddingMetadata,
    EmbeddingProvider,
    EmbeddingReader,
)
from rl4xs_wsi.data.embeddings.hdf5 import AtlasPatchHDF5Reader, HDF5EmbeddingProvider

__all__ = [
    "AtlasPatchHDF5Reader",
    "EmbeddingBatch",
    "EmbeddingFormatError",
    "EmbeddingMetadata",
    "EmbeddingProvider",
    "EmbeddingReader",
    "HDF5EmbeddingProvider",
]
