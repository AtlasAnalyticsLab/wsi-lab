"""Embedding format adapters."""

from wsi_lab.data.embeddings.base import (
    EmbeddingBatch,
    EmbeddingFormatError,
    EmbeddingMetadata,
    EmbeddingProvider,
    EmbeddingReader,
)
from wsi_lab.data.embeddings.hdf5 import AtlasPatchHDF5Reader, HDF5EmbeddingProvider

__all__ = [
    "AtlasPatchHDF5Reader",
    "EmbeddingBatch",
    "EmbeddingFormatError",
    "EmbeddingMetadata",
    "EmbeddingProvider",
    "EmbeddingReader",
    "HDF5EmbeddingProvider",
]
