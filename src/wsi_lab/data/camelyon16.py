"""CAMELYON16 discovery using its authoritative reference table."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from wsi_lab.constants import BELIEF_MAGNIFICATIONS, magnification_token
from wsi_lab.data.base import (
    DatasetAdapter,
    EmbeddingLocation,
    SlideRecord,
    reject_excluded_dataset,
)
from wsi_lab.errors import DatasetValidationError

_SLIDE_ID = re.compile(r"^(normal|tumor|test)_\d{3}$")
_LABELS = {"normal": 0, "tumor": 1}


class Camelyon16Adapter(DatasetAdapter):
    """Resolve the 399-slide CAMELYON16 cohort and AtlasPatch HDF5 files.

    Discovery is label-table driven.  This ensures that a missing WSI remains
    visible to validation rather than silently disappearing from a directory
    glob, and prevents the unlabeled ``test_*`` filename prefix from being
    treated as a class label.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        slides_dir: str | Path = "images",
        labels_path: str | Path = "evaluation/reference.csv",
        embeddings_root: str | Path = "Embeddings",
        tissue_masks_dir: str | Path = "background_tissue",
        tumor_masks_dir: str | Path = "masks",
        annotations_dir: str | Path = "annotations",
        encoder: str = "uni_v2",
        magnifications: tuple[float, ...] = BELIEF_MAGNIFICATIONS,
        directory_template: str = "UNI_v2_{magnification}x_256/patches",
        feature_key: str = "features/uni_v2",
        coordinate_key: str = "coords",
        passport_key: str | None = "passports",
    ) -> None:
        self._root = reject_excluded_dataset(root)
        self._slides_dir = self._resolve_under_root(slides_dir)
        self._labels_path = self._resolve_under_root(labels_path)
        self._embeddings_root = self._resolve_under_root(embeddings_root)
        if any(part.casefold() == "bench_embeddings" for part in self._embeddings_root.parts):
            raise ValueError(
                "bench_embeddings is non-canonical and cannot be used as embeddings.root"
            )
        self._tissue_masks_dir = self._resolve_under_root(tissue_masks_dir)
        self._tumor_masks_dir = self._resolve_under_root(tumor_masks_dir)
        self._annotations_dir = self._resolve_under_root(annotations_dir)
        self._encoder = encoder
        self._magnifications = tuple(float(value) for value in magnifications)
        self._directory_template = directory_template
        self._feature_key = feature_key.strip("/")
        self._coordinate_key = coordinate_key.strip("/")
        self._passport_key = passport_key.strip("/") if passport_key else None
        if not self._magnifications:
            raise ValueError("at least one embedding magnification is required")
        if len(set(self._magnifications)) != len(self._magnifications):
            raise ValueError("embedding magnifications must be unique")
        self._records = self._read_reference()

    @classmethod
    def from_config(cls, dataset: Any, embeddings: Any) -> Camelyon16Adapter:
        """Construct from duck-typed shared config models without coupling types."""

        return cls(
            dataset.root,
            slides_dir=dataset.slides_dir,
            labels_path=dataset.labels,
            embeddings_root=embeddings.root,
            tissue_masks_dir=dataset.tissue_masks_dir,
            tumor_masks_dir=dataset.tumor_masks_dir,
            annotations_dir=dataset.annotations_dir,
            encoder=embeddings.encoder,
            magnifications=tuple(embeddings.magnifications),
            directory_template=embeddings.directory_template,
            feature_key=embeddings.feature_key,
            coordinate_key=embeddings.coordinate_key,
            passport_key=embeddings.passport_key,
        )

    @property
    def name(self) -> str:
        return "camelyon16"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def magnifications(self) -> tuple[float, ...]:
        return self._magnifications

    @property
    def labels_path(self) -> Path:
        return self._labels_path

    @property
    def metadata_paths(self) -> tuple[Path, ...]:
        return (self._labels_path,)

    @property
    def embeddings_root(self) -> Path:
        return self._embeddings_root

    def iter_slides(self) -> Iterator[SlideRecord]:
        return iter(self._records.values())

    def get_slide(self, slide_id: str) -> SlideRecord:
        return self._records[slide_id]

    def embedding_location(self, slide_id: str, magnification: float) -> EmbeddingLocation:
        if slide_id not in self._records:
            raise KeyError(slide_id)
        value = float(magnification)
        if value not in self._magnifications:
            raise KeyError(f"magnification {value:g} is not configured")
        token = magnification_token(value)
        try:
            relative = self._directory_template.format(
                magnification=token,
                encoder=self._encoder,
                slide_id=slide_id,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"invalid embedding directory template: {self._directory_template!r}"
            ) from exc
        directory = Path(relative)
        if directory.is_absolute():
            raise ValueError("embedding directory_template must be relative to embeddings.root")
        if ".." in directory.parts:
            raise ValueError("embedding directory_template cannot contain '..' traversal")
        path = reject_excluded_dataset(self._embeddings_root / directory / f"{slide_id}.h5")
        if any(part.casefold() == "bench_embeddings" for part in path.parts):
            raise ValueError("bench_embeddings artifacts are explicitly excluded")
        if self._embeddings_root not in path.parents:
            raise ValueError("embedding directory_template escapes embeddings.root")
        return EmbeddingLocation(
            slide_id=slide_id,
            magnification=value,
            path=path,
            encoder=self._encoder,
            feature_key=self._feature_key,
            coordinate_key=self._coordinate_key,
            passport_key=self._passport_key,
        )

    def _resolve_under_root(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return reject_excluded_dataset(path if path.is_absolute() else self._root / path)

    def _read_reference(self) -> dict[str, SlideRecord]:
        if not self._labels_path.is_file():
            raise DatasetValidationError(
                f"CAMELYON16 reference table does not exist: {self._labels_path}"
            )
        records: dict[str, SlideRecord] = {}
        with self._labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"image", "type", "class", "center"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise DatasetValidationError(
                    f"CAMELYON16 reference table is missing columns {sorted(missing)}: {self._labels_path}"
                )
            for line_number, row in enumerate(reader, start=2):
                image_name = (row.get("image") or "").strip()
                slide_id = Path(image_name).stem
                if not _SLIDE_ID.fullmatch(slide_id):
                    raise DatasetValidationError(
                        f"invalid CAMELYON16 slide ID {slide_id!r} at {self._labels_path}:{line_number}"
                    )
                if slide_id in records:
                    raise DatasetValidationError(f"duplicate CAMELYON16 label row for {slide_id}")
                label_name = (row.get("type") or "").strip().casefold()
                if label_name not in _LABELS:
                    raise DatasetValidationError(
                        f"invalid CAMELYON16 type {label_name!r} for {slide_id}; expected normal or tumor"
                    )
                tissue = self._tissue_masks_dir / f"{slide_id}_tissue.tif"
                tumor_mask = self._tumor_masks_dir / f"{slide_id}_mask.tif"
                annotation = self._annotations_dir / f"{slide_id}.xml"
                records[slide_id] = SlideRecord(
                    slide_id=slide_id,
                    slide_path=self._slides_dir / image_name,
                    label=_LABELS[label_name],
                    label_name=label_name,
                    official_cohort="test" if slide_id.startswith("test_") else "development",
                    patient_id=None,
                    center=(row.get("center") or "").strip() or None,
                    tumor_class=(row.get("class") or "").strip().casefold() or None,
                    tissue_mask_path=tissue if tissue.is_file() else None,
                    tumor_mask_path=tumor_mask if tumor_mask.is_file() else None,
                    annotation_path=annotation if annotation.is_file() else None,
                    metadata={"reference_image": image_name},
                )
        if not records:
            raise DatasetValidationError(
                f"CAMELYON16 reference table is empty: {self._labels_path}"
            )
        return records


__all__ = ["Camelyon16Adapter"]
