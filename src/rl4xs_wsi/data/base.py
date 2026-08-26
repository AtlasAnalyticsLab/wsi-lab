"""Format-neutral dataset contracts.

The environment and training code consume these records rather than relying on
CAMELYON16 filename conventions.  In particular, labels for ``test_*`` slides
must come from the reference table; they are never inferred from the prefix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SlideRecord:
    """One labeled whole-slide image and its optional sidecars."""

    slide_id: str
    slide_path: Path
    label: int
    label_name: str
    official_cohort: str
    patient_id: str | None = None
    center: str | None = None
    tumor_class: str | None = None
    tissue_mask_path: Path | None = None
    tumor_mask_path: Path | None = None
    annotation_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_official_test(self) -> bool:
        return self.official_cohort == "test"


@dataclass(frozen=True, slots=True)
class EmbeddingLocation:
    """Resolved location and dataset keys for a slide-scale feature artifact."""

    slide_id: str
    magnification: float
    path: Path
    encoder: str
    feature_key: str
    coordinate_key: str = "coords"
    passport_key: str | None = "passports"


class DatasetAdapter(ABC):
    """Read-only adapter from an on-disk cohort to normalized records."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable dataset identifier."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """Resolved dataset root."""

    @property
    @abstractmethod
    def magnifications(self) -> tuple[float, ...]:
        """Embedding magnifications required by this adapter instance."""

    @abstractmethod
    def iter_slides(self) -> Iterator[SlideRecord]:
        """Yield every expected labeled slide in deterministic order."""

    @abstractmethod
    def get_slide(self, slide_id: str) -> SlideRecord:
        """Return a record or raise ``KeyError`` for an unknown slide ID."""

    @abstractmethod
    def embedding_location(self, slide_id: str, magnification: float) -> EmbeddingLocation:
        """Resolve one required embedding artifact without opening it."""

    def slide_ids(self) -> tuple[str, ...]:
        return tuple(record.slide_id for record in self.iter_slides())

    def embedding_locations(self, slide_id: str) -> tuple[EmbeddingLocation, ...]:
        return tuple(self.embedding_location(slide_id, value) for value in self.magnifications)


def reject_excluded_dataset(path: str | Path) -> Path:
    """Resolve a path while enforcing the project's explicit PEAN exclusion."""

    resolved = Path(path).expanduser().resolve()
    if any(part.casefold() == "pean" for part in resolved.parts):
        raise ValueError("PEAN is explicitly out of scope and cannot be used as a dataset root")
    return resolved


__all__ = ["DatasetAdapter", "EmbeddingLocation", "SlideRecord", "reject_excluded_dataset"]
