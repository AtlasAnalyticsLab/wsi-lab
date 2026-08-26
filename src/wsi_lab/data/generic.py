"""Manifest-driven adapter for datasets without a dedicated native adapter.

The generic adapter deliberately treats the dataset manifest as authoritative.
It never guesses labels, cohorts, or patient membership from filenames.  All
paths stored in the manifest are relative to the dataset root so a manifest is
portable and cannot make discovery escape into unrelated filesystem content.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from wsi_lab.constants import BELIEF_MAGNIFICATIONS, magnification_token
from wsi_lab.data.base import (
    DatasetAdapter,
    EmbeddingLocation,
    SlideRecord,
    reject_excluded_dataset,
)
from wsi_lab.errors import DatasetValidationError

_REQUIRED_COLUMNS = frozenset(
    {
        "slide_id",
        "image_path",
        "label",
        "label_name",
        "patient_id",
        "center",
        "official_cohort",
    }
)
_OPTIONAL_PATH_COLUMNS = (
    "tissue_mask_path",
    "tumor_mask_path",
    "annotation_path",
)
_OFFICIAL_COHORTS = frozenset({"development", "test", "test_unlabeled"})
_SAFE_SLIDE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXCLUDED_EMBEDDING_PARTS = frozenset({"pean", "bench_embeddings"})


class GenericDatasetAdapter(DatasetAdapter):
    """Read a normalized ``metadata.csv`` for an otherwise unknown dataset.

    Images and optional sidecars are named explicitly in the CSV.  Embedding
    paths remain convention based so every configured scale resolves to
    ``embeddings.root / directory_template / <slide_id>.h5``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        labels_path: str | Path = "metadata.csv",
        embeddings_root: str | Path = "Embeddings",
        encoder: str = "uni_v2",
        magnifications: tuple[float, ...] = BELIEF_MAGNIFICATIONS,
        directory_template: str = "UNI_v2_{magnification}x_256/patches",
        feature_key: str = "features/uni_v2",
        coordinate_key: str = "coords",
        passport_key: str | None = "passports",
    ) -> None:
        self._root = reject_excluded_dataset(root)
        if _contains_excluded_part(self._root):
            raise ValueError("PEAN and bench_embeddings are explicitly excluded")

        self._labels_path = self._resolve_config_path(labels_path, field="dataset.labels")
        self._embeddings_root = self._resolve_config_path(
            embeddings_root,
            field="embeddings.root",
        )
        if _contains_excluded_part(self._embeddings_root):
            raise ValueError("PEAN and bench_embeddings are explicitly excluded")

        normalized_encoder = encoder.strip()
        if normalized_encoder.casefold() in _EXCLUDED_EMBEDDING_PARTS:
            raise ValueError("PEAN and bench_embeddings are explicitly excluded")
        if not normalized_encoder:
            raise ValueError("embedding encoder cannot be empty")
        self._encoder = normalized_encoder

        self._magnifications = tuple(float(value) for value in magnifications)
        if not self._magnifications:
            raise ValueError("at least one embedding magnification is required")
        if len(set(self._magnifications)) != len(self._magnifications):
            raise ValueError("embedding magnifications must be unique")
        if any(not math.isfinite(value) or value <= 0 for value in self._magnifications):
            raise ValueError("embedding magnifications must be finite and positive")

        if not directory_template.strip():
            raise ValueError("embedding directory_template cannot be empty")
        self._directory_template = directory_template
        self._feature_key = _normalize_hdf5_key(feature_key, "feature_key")
        self._coordinate_key = _normalize_hdf5_key(coordinate_key, "coordinate_key")
        self._passport_key = (
            _normalize_hdf5_key(passport_key, "passport_key") if passport_key else None
        )
        self._records = self._read_manifest()

    @classmethod
    def from_config(cls, dataset: Any, embeddings: Any) -> GenericDatasetAdapter:
        """Construct from duck-typed shared config models."""

        root = Path(dataset.root).expanduser().resolve()

        def relative_to_root(value: Any) -> Path:
            path = Path(value).expanduser()
            return path.resolve().relative_to(root) if path.is_absolute() else path

        return cls(
            root,
            labels_path=relative_to_root(dataset.labels),
            embeddings_root=relative_to_root(embeddings.root),
            encoder=embeddings.encoder,
            magnifications=tuple(embeddings.magnifications),
            directory_template=embeddings.directory_template,
            feature_key=embeddings.feature_key,
            coordinate_key=embeddings.coordinate_key,
            passport_key=embeddings.passport_key,
        )

    @property
    def name(self) -> str:
        return "generic"

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
            rendered = self._directory_template.format(
                magnification=token,
                encoder=self._encoder,
                slide_id=slide_id,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"invalid embedding directory template: {self._directory_template!r}"
            ) from exc
        directory = _validated_relative_path(
            rendered,
            field="embeddings.directory_template",
        )
        if _contains_excluded_part(directory):
            raise ValueError("PEAN and bench_embeddings are explicitly excluded")

        path = (self._embeddings_root / directory / f"{slide_id}.h5").resolve()
        if self._embeddings_root not in path.parents:
            raise ValueError("embedding directory_template escapes embeddings.root")
        if _contains_excluded_part(path):
            raise ValueError("PEAN and bench_embeddings are explicitly excluded")
        return EmbeddingLocation(
            slide_id=slide_id,
            magnification=value,
            path=path,
            encoder=self._encoder,
            feature_key=self._feature_key,
            coordinate_key=self._coordinate_key,
            passport_key=self._passport_key,
        )

    def _resolve_config_path(self, value: str | Path, *, field: str) -> Path:
        """Resolve config paths, including loader-normalized in-root absolutes."""

        raw = Path(value).expanduser()
        if raw.is_absolute():
            resolved = raw.resolve()
        else:
            relative = _validated_relative_path(value, field=field)
            resolved = (self._root / relative).resolve()
        if self._root not in resolved.parents:
            raise DatasetValidationError(f"{field} escapes dataset.root: {value!s}")
        return reject_excluded_dataset(resolved)

    def _resolve_manifest_path(self, value: str | Path, *, field: str) -> Path:
        """Resolve an explicitly root-relative path read from metadata.csv."""

        relative = _validated_relative_path(value, field=field)
        resolved = (self._root / relative).resolve()
        if self._root not in resolved.parents:
            raise DatasetValidationError(f"{field} escapes dataset.root: {value!s}")
        return reject_excluded_dataset(resolved)

    def _read_manifest(self) -> dict[str, SlideRecord]:
        if not self._labels_path.is_file():
            raise DatasetValidationError(
                f"generic dataset manifest does not exist: {self._labels_path}"
            )

        records: dict[str, SlideRecord] = {}
        with self._labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = _REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise DatasetValidationError(
                    f"generic dataset manifest is missing columns {sorted(missing)}: "
                    f"{self._labels_path}"
                )
            for line_number, row in enumerate(reader, start=2):
                slide_id = (row.get("slide_id") or "").strip()
                if not _SAFE_SLIDE_ID.fullmatch(slide_id):
                    raise DatasetValidationError(
                        f"unsafe slide ID {slide_id!r} at {self._labels_path}:{line_number}"
                    )
                if slide_id in records:
                    raise DatasetValidationError(
                        f"duplicate slide ID {slide_id!r} in {self._labels_path}"
                    )

                cohort = (row.get("official_cohort") or "").strip().casefold()
                if cohort not in _OFFICIAL_COHORTS:
                    raise DatasetValidationError(
                        f"invalid official_cohort {cohort!r} for {slide_id}; expected one "
                        f"of {sorted(_OFFICIAL_COHORTS)}"
                    )
                label = self._parse_label(row.get("label"), cohort, slide_id, line_number)
                image_path = self._row_path(
                    row.get("image_path"),
                    field="image_path",
                    slide_id=slide_id,
                    line_number=line_number,
                    required=True,
                )
                assert image_path is not None

                optional_paths = {
                    field: self._row_path(
                        row.get(field),
                        field=field,
                        slide_id=slide_id,
                        line_number=line_number,
                        required=False,
                    )
                    for field in _OPTIONAL_PATH_COLUMNS
                }
                records[slide_id] = SlideRecord(
                    slide_id=slide_id,
                    slide_path=image_path,
                    label=label,
                    label_name=(row.get("label_name") or "").strip(),
                    official_cohort=cohort,
                    patient_id=(row.get("patient_id") or "").strip() or None,
                    center=(row.get("center") or "").strip() or None,
                    tissue_mask_path=optional_paths["tissue_mask_path"],
                    tumor_mask_path=optional_paths["tumor_mask_path"],
                    annotation_path=optional_paths["annotation_path"],
                    metadata={"manifest_line": line_number},
                )

        if not records:
            raise DatasetValidationError(f"generic dataset manifest is empty: {self._labels_path}")
        return records

    def _parse_label(
        self,
        raw_value: str | None,
        cohort: str,
        slide_id: str,
        line_number: int,
    ) -> int | None:
        value = (raw_value or "").strip()
        if not value:
            if cohort == "test_unlabeled":
                return None
            raise DatasetValidationError(
                f"blank label for {slide_id} at {self._labels_path}:{line_number}; "
                "only test_unlabeled rows may omit a label"
            )
        if value not in {"0", "1"}:
            raise DatasetValidationError(
                f"invalid label {value!r} for {slide_id}; expected integer 0 or 1"
            )
        return int(value)

    def _row_path(
        self,
        raw_value: str | None,
        *,
        field: str,
        slide_id: str,
        line_number: int,
        required: bool,
    ) -> Path | None:
        value = (raw_value or "").strip()
        if not value:
            if required:
                raise DatasetValidationError(
                    f"blank {field} for {slide_id} at {self._labels_path}:{line_number}"
                )
            return None
        try:
            return self._resolve_manifest_path(value, field=field)
        except (DatasetValidationError, ValueError) as exc:
            raise DatasetValidationError(
                f"unsafe {field} for {slide_id} at {self._labels_path}:{line_number}: {exc}"
            ) from exc


def _validated_relative_path(value: str | Path, *, field: str) -> Path:
    raw = str(value).strip()
    if not raw:
        raise DatasetValidationError(f"{field} cannot be empty")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise DatasetValidationError(f"{field} must be a relative path")
    if ".." in posix.parts or ".." in windows.parts:
        raise DatasetValidationError(f"{field} cannot contain parent traversal")
    if "\\" in raw:
        raise DatasetValidationError(f"{field} must use forward slashes")
    if posix == PurePosixPath("."):
        raise DatasetValidationError(f"{field} must reference a path below dataset.root")
    return Path(*posix.parts)


def _contains_excluded_part(path: str | Path) -> bool:
    return any(part.casefold() in _EXCLUDED_EMBEDDING_PARTS for part in Path(path).parts)


def _normalize_hdf5_key(value: str, field: str) -> str:
    normalized = value.strip("/")
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


# A descriptive alias for callers that name the adapter after its input format.
ManifestDatasetAdapter = GenericDatasetAdapter

__all__ = ["GenericDatasetAdapter", "ManifestDatasetAdapter"]
