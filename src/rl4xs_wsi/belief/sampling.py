"""Public compatibility module for partial-bag sampling contracts."""

from rl4xs_wsi.belief.sampler import (
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
