"""Native-layout adapter for the TCGA non-small-cell lung cancer cohort."""

from __future__ import annotations

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

_TCGA_SLIDE_ID = re.compile(
    r"^TCGA-(?P<tss>[A-Z0-9]{2})-(?P<participant>[A-Z0-9]{4})-(?=\S).+$",
    re.IGNORECASE | re.ASCII,
)
_COHORTS = (("luad", 0), ("lusc", 1))


@dataclass(frozen=True, slots=True)
class _Cohort:
    name: str
    label: int
    directory: Path
    embeddings_root: Path


class TCGANsclcAdapter(DatasetAdapter):
    """Discover TCGA-LUAD and TCGA-LUSC without reorganizing their files.

    The two disease directories are class cohorts, not pre-existing train/test
    splits.  Every slide is therefore marked as development data; the split
    manager is responsible for keeping all slides from a patient together.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        luad_dir: str | Path = "TCGA-LUAD",
        lusc_dir: str | Path = "TCGA-LUSC",
        images_dir: str | Path = "images",
        embeddings_root: str | Path = "Embeddings",
        encoder: str = "uni_v2",
        magnifications: tuple[float, ...] = BELIEF_MAGNIFICATIONS,
        directory_template: str = "UNI_v2_{magnification}x_256/patches",
        feature_key: str = "features/uni_v2",
        coordinate_key: str = "coords",
        passport_key: str | None = "passports",
    ) -> None:
        self._root = reject_excluded_dataset(root)
        self._images_relative = self._validate_relative(images_dir, "images_dir")
        embeddings_relative = self._validate_relative(embeddings_root, "embeddings.root")
        self._reject_bench_embeddings(embeddings_relative)

        directories = {
            "luad": self._resolve_cohort_dir(luad_dir, "luad_dir"),
            "lusc": self._resolve_cohort_dir(lusc_dir, "lusc_dir"),
        }
        if directories["luad"] == directories["lusc"]:
            raise ValueError("luad_dir and lusc_dir must resolve to different directories")
        self._cohorts = {
            name: _Cohort(
                name=name,
                label=label,
                directory=directories[name],
                embeddings_root=reject_excluded_dataset(directories[name] / embeddings_relative),
            )
            for name, label in _COHORTS
        }

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

        self._records, self._slide_cohorts = self._discover()

    @classmethod
    def from_config(cls, dataset: Any, embeddings: Any) -> TCGANsclcAdapter:
        """Construct from the shared duck-typed configuration models."""

        root = Path(dataset.root).expanduser().resolve()

        def relative_to_root(value: Any) -> Path:
            path = Path(value).expanduser()
            return path.resolve().relative_to(root) if path.is_absolute() else path

        return cls(
            root,
            luad_dir=relative_to_root(dataset.luad_dir),
            lusc_dir=relative_to_root(dataset.lusc_dir),
            images_dir=relative_to_root(getattr(dataset, "slides_dir", Path("images"))),
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
        return "tcga_nsclc"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def magnifications(self) -> tuple[float, ...]:
        return self._magnifications

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
        directory = self._validate_relative(rendered, "embedding directory_template")
        self._reject_bench_embeddings(directory)

        cohort = self._slide_cohorts[slide_id]
        path = reject_excluded_dataset(cohort.embeddings_root / directory / f"{slide_id}.h5")
        self._reject_bench_embeddings(path)
        if cohort.embeddings_root not in path.parents:
            raise ValueError("embedding directory_template escapes the cohort embeddings root")
        return EmbeddingLocation(
            slide_id=slide_id,
            magnification=value,
            path=path,
            encoder=self._encoder,
            feature_key=self._feature_key,
            coordinate_key=self._coordinate_key,
            passport_key=self._passport_key,
        )

    def _discover(self) -> tuple[dict[str, SlideRecord], dict[str, _Cohort]]:
        pending: list[tuple[str, SlideRecord, _Cohort]] = []
        seen_ids: dict[str, tuple[str, Path]] = {}
        patient_classes: dict[str, str] = {}

        for cohort_name, _label in _COHORTS:
            cohort = self._cohorts[cohort_name]
            image_root = cohort.directory / self._images_relative
            if not image_root.is_dir():
                raise DatasetValidationError(
                    f"TCGA-{cohort_name.upper()} image directory does not exist: {image_root}"
                )
            slide_paths = sorted(
                (
                    path
                    for path in image_root.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".svs"
                ),
                key=lambda path: (path.name.casefold(), path.name),
            )
            if not slide_paths:
                raise DatasetValidationError(
                    f"TCGA-{cohort_name.upper()} contains no .svs files: {image_root}"
                )

            for slide_path in slide_paths:
                slide_id = slide_path.stem
                match = _TCGA_SLIDE_ID.fullmatch(slide_id)
                if match is None:
                    raise DatasetValidationError(
                        f"invalid TCGA slide ID {slide_id!r}: {slide_path}"
                    )
                folded_id = slide_id.casefold()
                previous = seen_ids.get(folded_id)
                if previous is not None:
                    previous_cohort, previous_path = previous
                    raise DatasetValidationError(
                        "duplicate TCGA slide ID "
                        f"{slide_id!r} in {previous_cohort} ({previous_path}) and "
                        f"{cohort_name} ({slide_path})"
                    )
                seen_ids[folded_id] = (cohort_name, slide_path)

                tss = match.group("tss").upper()
                participant = match.group("participant").upper()
                patient_id = f"TCGA-{tss}-{participant}"
                previous_class = patient_classes.get(patient_id)
                if previous_class is not None and previous_class != cohort_name:
                    raise DatasetValidationError(
                        f"TCGA patient {patient_id} appears in conflicting classes "
                        f"{previous_class} and {cohort_name}"
                    )
                patient_classes[patient_id] = cohort_name

                record = SlideRecord(
                    slide_id=slide_id,
                    slide_path=slide_path,
                    label=cohort.label,
                    label_name=cohort_name,
                    official_cohort="development",
                    patient_id=patient_id,
                    center=tss,
                    tumor_class=cohort_name,
                    metadata={"source_cohort": f"TCGA-{cohort_name.upper()}"},
                )
                pending.append((folded_id, record, cohort))

        records: dict[str, SlideRecord] = {}
        slide_cohorts: dict[str, _Cohort] = {}
        for _folded_id, record, cohort in sorted(
            pending, key=lambda item: (item[0], item[1].slide_id)
        ):
            records[record.slide_id] = record
            slide_cohorts[record.slide_id] = cohort
        return records, slide_cohorts

    def _resolve_cohort_dir(self, value: str | Path, field: str) -> Path:
        relative = self._validate_relative(value, field)
        resolved = reject_excluded_dataset(self._root / relative)
        if self._root not in resolved.parents:
            raise ValueError(f"{field} escapes the dataset root")
        return resolved

    @staticmethod
    def _validate_relative(value: str | Path, field: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            raise ValueError(f"{field} must be relative")
        if ".." in path.parts:
            raise ValueError(f"{field} cannot contain '..' traversal")
        if not path.parts or str(path) in {"", "."}:
            raise ValueError(f"{field} cannot be empty")
        return path

    @staticmethod
    def _reject_bench_embeddings(path: Path) -> None:
        if any(part.casefold() == "bench_embeddings" for part in path.parts):
            raise ValueError("bench_embeddings artifacts are explicitly excluded")


__all__ = ["TCGANsclcAdapter"]
