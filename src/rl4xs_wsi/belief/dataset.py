"""Dynamic partial-bag datasets and padded ABMIL collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rl4xs_wsi.belief.sampler import (
    BagSelection,
    PartialBagSampler,
    PartialBagSamplingConfig,
    SlideEmbeddingRecord,
)
from rl4xs_wsi.constants import BELIEF_MAGNIFICATIONS
from rl4xs_wsi.errors import TestSplitAccessError

_TRAIN_PARTITIONS = frozenset({"train", "training"})
_VALIDATION_PARTITIONS = frozenset({"val", "valid", "validation", "dev", "development"})


def _partition_token(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _is_test_partition(value: Any) -> bool:
    token = _partition_token(value)
    return token == "test" or "test" in token.split("_")


def validate_slide_partition(
    slides: Sequence[SlideEmbeddingRecord],
    *,
    expected: str,
) -> None:
    """Reject test leakage and enforce explicit split markers when present."""

    if expected not in {"train", "validation"}:
        raise ValueError(f"Unknown belief partition {expected!r}")
    allowed = _TRAIN_PARTITIONS if expected == "train" else _VALIDATION_PARTITIONS
    for slide in slides:
        metadata = slide.metadata
        if bool(metadata.get("is_test", False)) or bool(metadata.get("official_test", False)):
            raise TestSplitAccessError(
                f"Belief {expected} data contains official test slide {slide.slide_id!r}"
            )
        for key in ("cohort", "split", "partition"):
            value = metadata.get(key)
            if value is not None and _is_test_partition(value):
                raise TestSplitAccessError(
                    f"Belief {expected} data contains test slide {slide.slide_id!r} "
                    f"({key}={value!r})"
                )
        # A CAMELYON16 validation slide can legitimately retain cohort=train;
        # only split/partition describe its model-selection role.
        explicit = metadata.get("split", metadata.get("partition"))
        if explicit is not None and _partition_token(explicit) not in allowed:
            raise TestSplitAccessError(
                f"Slide {slide.slide_id!r} is marked split={explicit!r}, but was "
                f"provided to the belief {expected} partition"
            )


@dataclass(frozen=True, slots=True)
class BeliefBag:
    """One unpadded mixed-scale bag."""

    features: Tensor
    magnifications: Tensor
    label: Tensor
    slide_id: str
    requested_size: int

    @property
    def bag_size(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True, slots=True)
class BeliefBatch:
    """A padded batch with an explicit validity mask."""

    features: Tensor
    magnifications: Tensor
    mask: Tensor
    labels: Tensor
    slide_ids: tuple[str, ...]
    bag_sizes: Tensor
    requested_sizes: Tensor

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> BeliefBatch:
        return BeliefBatch(
            features=self.features.to(device, non_blocking=non_blocking),
            magnifications=self.magnifications.to(device, non_blocking=non_blocking),
            mask=self.mask.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            slide_ids=self.slide_ids,
            bag_sizes=self.bag_sizes.to(device, non_blocking=non_blocking),
            requested_sizes=self.requested_sizes.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> BeliefBatch:
        return BeliefBatch(
            features=self.features.pin_memory(),
            magnifications=self.magnifications.pin_memory(),
            mask=self.mask.pin_memory(),
            labels=self.labels.pin_memory(),
            slide_ids=self.slide_ids,
            bag_sizes=self.bag_sizes.pin_memory(),
            requested_sizes=self.requested_sizes.pin_memory(),
        )


def _read_rows(matrix: Any, indices: np.ndarray) -> Tensor:
    if isinstance(matrix, Tensor):
        tensor_indices = torch.as_tensor(indices, dtype=torch.long, device=matrix.device)
        return matrix.index_select(0, tensor_indices).detach().to(device="cpu", dtype=torch.float32)
    rows = np.asarray(matrix[indices])
    if rows.ndim != 2:
        raise ValueError(f"Feature provider returned shape {rows.shape}; expected [N,D]")
    # Copying ensures tensors do not retain a read-only mmap/HDF5 buffer.
    return torch.from_numpy(np.array(rows, dtype=np.float32, copy=True))


def materialize_selection(
    slide: SlideEmbeddingRecord,
    selection: BagSelection,
    rng: np.random.Generator,
) -> BeliefBag:
    """Read selected rows lazily and randomize their order within the bag."""

    feature_parts: list[Tensor] = []
    magnification_parts: list[Tensor] = []
    for magnification in sorted(selection.indices_by_magnification):
        indices = selection.indices_by_magnification[magnification]
        rows = _read_rows(slide.features_by_magnification[magnification], indices)
        feature_parts.append(rows)
        magnification_parts.append(torch.full((rows.shape[0],), magnification, dtype=torch.float32))
    if not feature_parts:
        raise ValueError(f"Cannot materialize an empty bag for slide {slide.slide_id!r}")
    features = torch.cat(feature_parts, dim=0)
    magnifications = torch.cat(magnification_parts, dim=0)
    permutation = torch.as_tensor(rng.permutation(features.shape[0]), dtype=torch.long)
    return BeliefBag(
        features=features.index_select(0, permutation),
        magnifications=magnifications.index_select(0, permutation),
        label=torch.tensor(float(slide.label), dtype=torch.float32),
        slide_id=slide.slide_id,
        requested_size=selection.requested_size,
    )


def collate_belief_bags(bags: Sequence[BeliefBag]) -> BeliefBatch:
    """Pad variable-length bags without creating false scale encodings."""

    if not bags:
        raise ValueError("Cannot collate an empty bag list")
    dimensions = {int(bag.features.shape[1]) for bag in bags}
    if len(dimensions) != 1:
        raise ValueError(f"Cannot collate inconsistent embedding dimensions: {sorted(dimensions)}")
    maximum = max(bag.bag_size for bag in bags)
    batch_size = len(bags)
    embedding_dim = dimensions.pop()
    features = torch.zeros((batch_size, maximum, embedding_dim), dtype=torch.float32)
    # Padding carries a valid protocol scale so one-hot conversion can happen
    # before masking.  Its feature and attention contribution is masked out.
    magnifications = torch.full(
        (batch_size, maximum), BELIEF_MAGNIFICATIONS[0], dtype=torch.float32
    )
    mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    labels = torch.empty((batch_size,), dtype=torch.float32)
    bag_sizes = torch.empty((batch_size,), dtype=torch.long)
    requested_sizes = torch.empty((batch_size,), dtype=torch.long)
    slide_ids: list[str] = []

    for row, bag in enumerate(bags):
        size = bag.bag_size
        if size < 1:
            raise ValueError(f"Slide {bag.slide_id!r} supplied an empty bag")
        features[row, :size].copy_(bag.features.to(dtype=torch.float32))
        magnifications[row, :size].copy_(bag.magnifications.to(dtype=torch.float32))
        mask[row, :size] = True
        labels[row] = bag.label
        bag_sizes[row] = size
        requested_sizes[row] = bag.requested_size
        slide_ids.append(bag.slide_id)
    return BeliefBatch(
        features=features,
        magnifications=magnifications,
        mask=mask,
        labels=labels,
        slide_ids=tuple(slide_ids),
        bag_sizes=bag_sizes,
        requested_sizes=requested_sizes,
    )


def _rng(seed: int, *components: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), *(int(value) for value in components)])
    return np.random.default_rng(sequence)


class PartialBagDataset(Dataset[BeliefBag]):
    """Dynamic training bags, reproducible by seed, epoch and sample index."""

    def __init__(
        self,
        slides: Sequence[SlideEmbeddingRecord],
        *,
        sampler: PartialBagSampler | None = None,
        bags_per_slide_per_epoch: int = 8,
        seed: int = 1,
    ) -> None:
        self.slides = tuple(slides)
        if not self.slides:
            raise ValueError("PartialBagDataset requires at least one slide")
        dimensions = {slide.embedding_dim for slide in self.slides}
        if len(dimensions) != 1:
            raise ValueError(f"Slides have inconsistent embedding dimensions: {sorted(dimensions)}")
        if bags_per_slide_per_epoch < 1:
            raise ValueError("bags_per_slide_per_epoch must be positive")
        self.sampler = sampler or PartialBagSampler()
        self.bags_per_slide_per_epoch = int(bags_per_slide_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self._slide_order = np.arange(len(self.slides), dtype=np.int64)
        self.set_epoch(0)

    @classmethod
    def from_config(
        cls,
        slides: Sequence[SlideEmbeddingRecord],
        config: Any,
    ) -> PartialBagDataset:
        get = (
            config.get
            if isinstance(config, Mapping)
            else lambda key, default: getattr(config, key, default)
        )
        sampling = PartialBagSamplingConfig.from_config(config)
        return cls(
            slides,
            sampler=PartialBagSampler(sampling),
            bags_per_slide_per_epoch=int(cast(Any, get("bags_per_slide_per_epoch", 8))),
            seed=int(cast(Any, get("seed", 1))),
        )

    @property
    def embedding_dim(self) -> int:
        return self.slides[0].embedding_dim

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)
        # Keep all bags from one slide contiguous so a DataLoader batch reads
        # one small set of HDF5 files while their chunks are hot in the OS
        # cache. The slide order is still deterministically reshuffled every
        # epoch, and every slide contributes exactly the configured bag count.
        self._slide_order = _rng(self.seed, 0x534C4944, self.epoch).permutation(len(self.slides))

    def __len__(self) -> int:
        return len(self.slides) * self.bags_per_slide_per_epoch

    def __getitem__(self, index: int) -> BeliefBag:
        if not 0 <= index < len(self):
            raise IndexError(index)
        slide_position, replicate = divmod(index, self.bags_per_slide_per_epoch)
        slide_index = int(self._slide_order[slide_position])
        slide = self.slides[slide_index]
        random = _rng(self.seed, 0x54524149, self.epoch, slide_index, replicate)
        selection = self.sampler.sample(slide, random)
        return materialize_selection(slide, selection, random)


class ValidationBagDataset(Dataset[BeliefBag]):
    """Fixed validation bags indexed by slide, K and deterministic replicate."""

    def __init__(
        self,
        slides: Sequence[SlideEmbeddingRecord],
        *,
        bag_sizes: Sequence[int] = (1, 4, 8, 16, 32, 64, 128, 256, 500),
        bags_per_slide: int = 8,
        max_bag_size: int = 500,
        seed: int = 1,
    ) -> None:
        self.slides = tuple(slides)
        if not self.slides:
            raise ValueError("ValidationBagDataset requires at least one slide")
        dimensions = {slide.embedding_dim for slide in self.slides}
        if len(dimensions) != 1:
            raise ValueError(f"Slides have inconsistent embedding dimensions: {sorted(dimensions)}")
        self.bag_sizes = tuple(int(size) for size in bag_sizes)
        if not self.bag_sizes or any(size < 1 or size > max_bag_size for size in self.bag_sizes):
            raise ValueError("validation bag sizes must be in [1, max_bag_size]")
        if bags_per_slide < 1:
            raise ValueError("bags_per_slide must be positive")
        self.bags_per_slide = int(bags_per_slide)
        self.seed = int(seed)
        self.sampler = PartialBagSampler(PartialBagSamplingConfig(max_bag_size=int(max_bag_size)))

    @classmethod
    def from_config(
        cls,
        slides: Sequence[SlideEmbeddingRecord],
        config: Any,
    ) -> ValidationBagDataset:
        get = (
            config.get
            if isinstance(config, Mapping)
            else lambda key, default: getattr(config, key, default)
        )
        return cls(
            slides,
            bag_sizes=cast(
                Sequence[int],
                get("validation_bag_sizes", (1, 4, 8, 16, 32, 64, 128, 256, 500)),
            ),
            bags_per_slide=int(cast(Any, get("validation_bags_per_slide", 8))),
            max_bag_size=int(cast(Any, get("max_bag_size", 500))),
            seed=int(cast(Any, get("seed", 1))),
        )

    def __len__(self) -> int:
        return len(self.slides) * len(self.bag_sizes) * self.bags_per_slide

    def __getitem__(self, index: int) -> BeliefBag:
        if not 0 <= index < len(self):
            raise IndexError(index)
        per_slide = len(self.bag_sizes) * self.bags_per_slide
        slide_index, within_slide = divmod(index, per_slide)
        size_index, replicate = divmod(within_slide, self.bags_per_slide)
        slide = self.slides[slide_index]
        requested_size = self.bag_sizes[size_index]
        random = _rng(self.seed, 0x56414C49, slide_index, requested_size, replicate)
        selection = self.sampler.sample(slide, random, bag_size=requested_size)
        return materialize_selection(slide, selection, random)


class FullBagDataset(Dataset[BeliefBag]):
    """One deterministic complete embedding bag per slide for evaluation."""

    def __init__(self, slides: Sequence[SlideEmbeddingRecord], *, seed: int = 1) -> None:
        self.slides = tuple(slides)
        if not self.slides:
            raise ValueError("FullBagDataset requires at least one slide")
        dimensions = {slide.embedding_dim for slide in self.slides}
        if len(dimensions) != 1:
            raise ValueError(f"Slides have inconsistent embedding dimensions: {sorted(dimensions)}")
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, index: int) -> BeliefBag:
        if not 0 <= index < len(self):
            raise IndexError(index)
        slide = self.slides[index]
        selection = BagSelection(
            slide_id=slide.slide_id,
            label=slide.label,
            requested_size=slide.total_instances,
            indices_by_magnification={
                magnification: np.arange(matrix.shape[0], dtype=np.int64)
                for magnification, matrix in slide.features_by_magnification.items()
                if matrix.shape[0]
            },
        )
        return materialize_selection(slide, selection, _rng(self.seed, 0x46554C4C, index))


__all__ = [
    "BeliefBag",
    "BeliefBatch",
    "FullBagDataset",
    "PartialBagDataset",
    "ValidationBagDataset",
    "collate_belief_bags",
    "materialize_selection",
    "validate_slide_partition",
]
