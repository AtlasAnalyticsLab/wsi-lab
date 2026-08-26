"""Typed adapter selection for supported native dataset layouts."""

from __future__ import annotations

from typing import Any

from wsi_lab.data.base import DatasetAdapter


def create_dataset_adapter(dataset: Any, embeddings: Any) -> DatasetAdapter:
    """Construct the configured adapter without leaking dataset logic outward."""

    name = str(dataset.name)
    if name == "camelyon16":
        from wsi_lab.data.camelyon16 import Camelyon16Adapter

        return Camelyon16Adapter.from_config(dataset, embeddings)
    if name == "camelyon17":
        from wsi_lab.data.camelyon17 import Camelyon17Adapter

        return Camelyon17Adapter.from_config(dataset, embeddings)
    if name == "tcga_nsclc":
        from wsi_lab.data.tcga_nsclc import TCGANsclcAdapter

        return TCGANsclcAdapter.from_config(dataset, embeddings)
    if name == "generic":
        from wsi_lab.data.generic import GenericDatasetAdapter

        return GenericDatasetAdapter.from_config(dataset, embeddings)
    raise ValueError(f"unsupported dataset adapter: {name!r}")


__all__ = ["create_dataset_adapter"]
