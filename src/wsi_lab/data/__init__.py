"""Dataset discovery, embedding access, validation, and split management."""

from wsi_lab.data.base import DatasetAdapter, EmbeddingLocation, SlideRecord
from wsi_lab.data.camelyon16 import Camelyon16Adapter
from wsi_lab.data.camelyon16_17 import CombinedCamelyonAdapter
from wsi_lab.data.camelyon17 import Camelyon17Adapter
from wsi_lab.data.factory import create_dataset_adapter
from wsi_lab.data.generic import GenericDatasetAdapter, ManifestDatasetAdapter
from wsi_lab.data.splits import (
    DatasetSplit,
    SplitAssignment,
    UnassignedSlide,
    create_dataset_split,
    validate_dataset_split,
)
from wsi_lab.data.tcga_nsclc import TCGANsclcAdapter
from wsi_lab.data.validation import DatasetValidator, ValidationOptions, report_has_errors

__all__ = [
    "Camelyon16Adapter",
    "Camelyon17Adapter",
    "CombinedCamelyonAdapter",
    "DatasetAdapter",
    "DatasetSplit",
    "DatasetValidator",
    "EmbeddingLocation",
    "GenericDatasetAdapter",
    "ManifestDatasetAdapter",
    "SlideRecord",
    "SplitAssignment",
    "TCGANsclcAdapter",
    "UnassignedSlide",
    "ValidationOptions",
    "create_dataset_adapter",
    "create_dataset_split",
    "report_has_errors",
    "validate_dataset_split",
]
