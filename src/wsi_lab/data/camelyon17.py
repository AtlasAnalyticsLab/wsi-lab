"""CAMELYON17 discovery from its native image directory and stage table."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from dataclasses import dataclass
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

_SLIDE_ID = re.compile(r"^patient_(?P<patient>\d{3})_node_(?P<node>[0-4])$")
_PATIENT_SUMMARY = re.compile(r"^patient_\d{3}\.zip$", re.IGNORECASE)
_WSI_EXTENSIONS = frozenset({".bif", ".mrxs", ".ndpi", ".scn", ".svs", ".tif", ".tiff"})
_LABELS = {"negative": 0, "itc": 1, "micro": 1, "macro": 1}


@dataclass(frozen=True, slots=True)
class _StageRow:
    """Validated slide-level row from ``stages.csv``."""

    image_name: str
    label_name: str
    center: str | None


class Camelyon17Adapter(DatasetAdapter):
    """Expose native CAMELYON17 files through the shared dataset contract.

    CAMELYON17's stage table labels the first 100 patients. Images without a
    slide-level stage row are retained as the official unlabeled test cohort;
    labels are never inferred from patient or node numbers.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        slides_dir: str | Path = "images",
        stages_path: str | Path = "stages.csv",
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
        self._stages_path = self._resolve_under_root(stages_path)
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
        self._records = self._discover_records()

    @classmethod
    def from_config(cls, dataset: Any, embeddings: Any) -> Camelyon17Adapter:
        """Construct from duck-typed shared config models without coupling types."""

        return cls(
            dataset.root,
            slides_dir=dataset.slides_dir,
            stages_path=dataset.labels,
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
        return "camelyon17"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def magnifications(self) -> tuple[float, ...]:
        return self._magnifications

    @property
    def labels_path(self) -> Path:
        return self._stages_path

    @property
    def metadata_paths(self) -> tuple[Path, ...]:
        return (self._stages_path,)

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

    def _discover_records(self) -> dict[str, SlideRecord]:
        images = self._discover_images()
        stages = self._read_stages(images)
        records: dict[str, SlideRecord] = {}
        for slide_id, slide_path in images.items():
            match = _SLIDE_ID.fullmatch(slide_id)
            # ``_discover_images`` has already validated this; the guard keeps
            # the invariant explicit for static analysis and future changes.
            if match is None:  # pragma: no cover
                raise DatasetValidationError(f"invalid CAMELYON17 slide ID {slide_id!r}")
            patient_id = f"patient_{match.group('patient')}"
            patient_number = int(match.group("patient"))
            stage = stages.get(slide_id)
            tissue = self._tissue_masks_dir / f"{slide_id}_tissue.tif"
            tumor_mask = self._tumor_masks_dir / f"{slide_id}_mask.tif"
            annotation = self._annotations_dir / f"{slide_id}.xml"
            if stage is None:
                if patient_number <= 99:
                    raise DatasetValidationError(
                        f"CAMELYON17 development slide {slide_id!r} has no stage row in "
                        f"{self._stages_path}"
                    )
                label = None
                label_name = "unlabeled"
                official_cohort = "test_unlabeled"
                center = None
                tumor_class = None
                metadata: dict[str, Any] = {
                    "patient_number": patient_number,
                    "node_number": int(match.group("node")),
                }
            else:
                label = _LABELS[stage.label_name]
                label_name = stage.label_name
                official_cohort = "development"
                center = stage.center
                tumor_class = stage.label_name if label else None
                metadata = {
                    "stage_image": stage.image_name,
                    "patient_number": patient_number,
                    "node_number": int(match.group("node")),
                }
            records[slide_id] = SlideRecord(
                slide_id=slide_id,
                slide_path=slide_path,
                label=label,
                label_name=label_name,
                official_cohort=official_cohort,
                patient_id=patient_id,
                center=center,
                tumor_class=tumor_class,
                tissue_mask_path=tissue if tissue.is_file() else None,
                tumor_mask_path=tumor_mask if tumor_mask.is_file() else None,
                annotation_path=annotation if annotation.is_file() else None,
                metadata=metadata,
            )
        return records

    def _discover_images(self) -> dict[str, Path]:
        if not self._slides_dir.is_dir():
            raise DatasetValidationError(
                f"CAMELYON17 image directory does not exist: {self._slides_dir}"
            )
        images: dict[str, Path] = {}
        candidates = sorted(
            (
                path
                for path in self._slides_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in _WSI_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
        for path in candidates:
            slide_id = path.stem
            if _SLIDE_ID.fullmatch(slide_id) is None:
                raise DatasetValidationError(
                    f"invalid CAMELYON17 slide filename {path.name!r} in {self._slides_dir}"
                )
            if slide_id in images:
                raise DatasetValidationError(
                    f"duplicate CAMELYON17 image files for slide {slide_id!r}"
                )
            images[slide_id] = path
        if not images:
            raise DatasetValidationError(
                f"CAMELYON17 image directory contains no supported WSIs: {self._slides_dir}"
            )
        return images

    def _read_stages(self, images: dict[str, Path]) -> dict[str, _StageRow]:
        if not self._stages_path.is_file():
            raise DatasetValidationError(
                f"CAMELYON17 stage table does not exist: {self._stages_path}"
            )
        stages: dict[str, _StageRow] = {}
        with self._stages_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"patient", "stage", "center"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise DatasetValidationError(
                    f"CAMELYON17 stage table is missing columns {sorted(missing)}: "
                    f"{self._stages_path}"
                )
            for line_number, row in enumerate(reader, start=2):
                image_name = (row.get("patient") or "").strip()
                if _PATIENT_SUMMARY.fullmatch(image_name):
                    continue
                slide_id = Path(image_name).stem
                match = _SLIDE_ID.fullmatch(slide_id)
                if match is None or Path(image_name).suffix.casefold() not in _WSI_EXTENSIONS:
                    raise DatasetValidationError(
                        f"invalid CAMELYON17 stage image {image_name!r} at "
                        f"{self._stages_path}:{line_number}"
                    )
                if int(match.group("patient")) > 99:
                    raise DatasetValidationError(
                        f"CAMELYON17 stage row labels non-development patient {slide_id!r} at "
                        f"{self._stages_path}:{line_number}"
                    )
                if slide_id in stages:
                    raise DatasetValidationError(f"duplicate CAMELYON17 stage row for {slide_id!r}")
                if slide_id not in images:
                    raise DatasetValidationError(
                        f"CAMELYON17 stage row has no matching image for {slide_id!r}"
                    )
                label_name = (row.get("stage") or "").strip().casefold()
                if label_name not in _LABELS:
                    raise DatasetValidationError(
                        f"invalid CAMELYON17 stage {label_name!r} for {slide_id}; expected one of "
                        f"{sorted(_LABELS)}"
                    )
                stages[slide_id] = _StageRow(
                    image_name=image_name,
                    label_name=label_name,
                    center=(row.get("center") or "").strip() or None,
                )
        return stages


__all__ = ["Camelyon17Adapter"]
