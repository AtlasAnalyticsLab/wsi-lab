"""Public compatibility module for partial-bag sampling contracts."""

from wsi_lab.belief.sampler import (
    BagSelection,
    PartialBagSampler,
    PartialBagSamplingConfig,
    RowAccessibleMatrix,
    SlideEmbeddingRecord,
)

__all__ = [
    "BagSelection",
    "PartialBagSampler",
    "PartialBagSamplingConfig",
    "RowAccessibleMatrix",
    "SlideEmbeddingRecord",
]
