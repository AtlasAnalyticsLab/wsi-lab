"""Dataset discovery, embedding access, validation, and split management."""

from rl4xs_wsi.data.base import DatasetAdapter, EmbeddingLocation, SlideRecord
from rl4xs_wsi.data.camelyon16 import Camelyon16Adapter
from rl4xs_wsi.data.splits import (
    DatasetSplit,
    SplitAssignment,
    create_dataset_split,
    validate_dataset_split,
)
from rl4xs_wsi.data.validation import DatasetValidator, ValidationOptions, report_has_errors

__all__ = [
    "Camelyon16Adapter",
    "DatasetAdapter",
    "DatasetSplit",
    "DatasetValidator",
    "EmbeddingLocation",
    "SlideRecord",
    "SplitAssignment",
    "ValidationOptions",
    "create_dataset_split",
    "report_has_errors",
    "validate_dataset_split",
]
