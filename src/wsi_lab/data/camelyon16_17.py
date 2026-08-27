"""Read-only composition of the native CAMELYON16 and CAMELYON17 cohorts."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from wsi_lab.constants import BELIEF_MAGNIFICATIONS
from wsi_lab.data.base import (
    DatasetAdapter,
    EmbeddingLocation,
    SlideRecord,
    reject_excluded_dataset,
)
from wsi_lab.data.camelyon16 import Camelyon16Adapter
from wsi_lab.data.camelyon17 import Camelyon17Adapter
from wsi_lab.data.splits import DatasetSplit, validate_dataset_split
from wsi_lab.errors import DatasetValidationError


def _relative_layout_path(value: str | Path, root: Path, *, field: str) -> Path:
    """Recover one shared native-layout path after configuration resolution."""

    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"{field} must be relative to the combined dataset root so it can be "
                "applied beneath both source roots"
            ) from exc
    if ".." in path.parts:
        raise ValueError(f"{field} must be a relative path without '..' traversal")
    return path


def _source_root(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return reject_excluded_dataset(path if path.is_absolute() else root / path)


class CombinedCamelyonAdapter(DatasetAdapter):
    """Expose CAMELYON16 and CAMELYON17 as one binary classification cohort.

    Native slide IDs and artifacts remain unchanged. Ownership is recorded in
    ``metadata['source_dataset']`` and used to delegate embedding lookup to the
    correct source adapter. The two native split policies are executed
    independently and corresponding partitions are then unioned.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        camelyon16_root: str | Path = "CAMELYON16",
        camelyon17_root: str | Path = "CAMELYON17",
        slides_dir: str | Path = "images",
        camelyon16_labels_path: str | Path = "evaluation/reference.csv",
        camelyon17_labels_path: str | Path = "stages.csv",
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
        source16_root = _source_root(camelyon16_root, self._root)
        source17_root = _source_root(camelyon17_root, self._root)
        if source16_root == source17_root:
            raise ValueError("CAMELYON16 and CAMELYON17 source roots must be different")

        self._camelyon16 = Camelyon16Adapter(
            source16_root,
            slides_dir=slides_dir,
            labels_path=camelyon16_labels_path,
            embeddings_root=embeddings_root,
            tissue_masks_dir=tissue_masks_dir,
            tumor_masks_dir=tumor_masks_dir,
            annotations_dir=annotations_dir,
            encoder=encoder,
            magnifications=magnifications,
            directory_template=directory_template,
            feature_key=feature_key,
            coordinate_key=coordinate_key,
            passport_key=passport_key,
        )
        self._camelyon17 = Camelyon17Adapter(
            source17_root,
            slides_dir=slides_dir,
            stages_path=camelyon17_labels_path,
            embeddings_root=embeddings_root,
            tissue_masks_dir=tissue_masks_dir,
            tumor_masks_dir=tumor_masks_dir,
            annotations_dir=annotations_dir,
            encoder=encoder,
            magnifications=magnifications,
            directory_template=directory_template,
            feature_key=feature_key,
            coordinate_key=coordinate_key,
            passport_key=passport_key,
        )
        if self._camelyon16.magnifications != self._camelyon17.magnifications:
            raise ValueError("source adapters must expose identical embedding magnifications")

        records: dict[str, SlideRecord] = {}
        owners: dict[str, DatasetAdapter] = {}
        for source in (self._camelyon16, self._camelyon17):
            for record in source.iter_slides():
                if record.slide_id in records:
                    raise DatasetValidationError(
                        f"duplicate slide ID across CAMELYON sources: {record.slide_id!r}"
                    )
                records[record.slide_id] = replace(
                    record,
                    metadata={**record.metadata, "source_dataset": source.name},
                )
                owners[record.slide_id] = source
        self._records = dict(sorted(records.items()))
        self._owners = owners

    @classmethod
    def from_config(cls, dataset: Any, embeddings: Any) -> CombinedCamelyonAdapter:
        """Construct from the shared config while applying one layout to both roots."""

        root = Path(dataset.root).expanduser().resolve()
        return cls(
            root,
            camelyon16_root=dataset.camelyon16_root,
            camelyon17_root=dataset.camelyon17_root,
            slides_dir=_relative_layout_path(dataset.slides_dir, root, field="dataset.slides_dir"),
            camelyon16_labels_path=dataset.labels,
            camelyon17_labels_path=dataset.camelyon17_labels,
            embeddings_root=_relative_layout_path(embeddings.root, root, field="embeddings.root"),
            tissue_masks_dir=_relative_layout_path(
                dataset.tissue_masks_dir, root, field="dataset.tissue_masks_dir"
            ),
            tumor_masks_dir=_relative_layout_path(
                dataset.tumor_masks_dir, root, field="dataset.tumor_masks_dir"
            ),
            annotations_dir=_relative_layout_path(
                dataset.annotations_dir, root, field="dataset.annotations_dir"
            ),
            encoder=embeddings.encoder,
            magnifications=tuple(embeddings.magnifications),
            directory_template=embeddings.directory_template,
            feature_key=embeddings.feature_key,
            coordinate_key=embeddings.coordinate_key,
            passport_key=embeddings.passport_key,
        )

    @property
    def name(self) -> str:
        return "camelyon16_17"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def magnifications(self) -> tuple[float, ...]:
        return self._camelyon16.magnifications

    @property
    def metadata_paths(self) -> tuple[Path, ...]:
        return (*self._camelyon16.metadata_paths, *self._camelyon17.metadata_paths)

    @property
    def source_adapters(self) -> tuple[DatasetAdapter, DatasetAdapter]:
        return (self._camelyon16, self._camelyon17)

    def iter_slides(self) -> Iterator[SlideRecord]:
        return iter(self._records.values())

    def get_slide(self, slide_id: str) -> SlideRecord:
        return self._records[slide_id]

    def embedding_location(self, slide_id: str, magnification: float) -> EmbeddingLocation:
        return self._owners[slide_id].embedding_location(slide_id, magnification)

    def create_split(
        self,
        *,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 1,
        stratify_by_center: bool = True,
        prefer_patient_level: bool = True,
    ) -> DatasetSplit:
        """Union corresponding partitions from independently split native cohorts."""

        component_splits = tuple(
            source.create_split(
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=seed,
                stratify_by_center=stratify_by_center,
                prefer_patient_level=prefer_patient_level,
            )
            for source in self.source_adapters
        )
        assignments = tuple(
            sorted(
                (item for split in component_splits for item in split.assignments),
                key=lambda item: item.slide_id,
            )
        )
        unassigned = tuple(
            sorted(
                (item for split in component_splits for item in split.unassigned),
                key=lambda item: item.slide_id,
            )
        )
        represented = [item.slide_id for item in assignments]
        represented.extend(item.slide_id for item in unassigned)
        if len(represented) != len(set(represented)):
            raise DatasetValidationError("CAMELYON source splits contain duplicate slide IDs")
        result = DatasetSplit(
            dataset=self.name,
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            strategy="combined_corresponding",
            assignments=assignments,
            unassigned=unassigned,
            schema_version=2,
        )
        validate_dataset_split(result, expected_records=tuple(self.iter_slides()))
        return result


__all__ = ["CombinedCamelyonAdapter"]
