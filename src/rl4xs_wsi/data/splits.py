"""Leakage-safe, persisted slide or patient splits."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from rl4xs_wsi.data.base import SlideRecord
from rl4xs_wsi.errors import DatasetValidationError
from rl4xs_wsi.utils.atomic import atomic_json_dump
from rl4xs_wsi.utils.fingerprints import fingerprint

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    slide_id: str
    label: int
    split: SplitName
    patient_id: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    dataset: str
    seed: int
    validation_fraction: float
    strategy: str
    assignments: tuple[SplitAssignment, ...]
    schema_version: int = 1

    def ids(self, split: SplitName) -> tuple[str, ...]:
        return tuple(item.slide_id for item in self.assignments if item.split == split)

    def assignment(self, slide_id: str) -> SplitAssignment:
        for item in self.assignments:
            if item.slide_id == slide_id:
                return item
        raise KeyError(slide_id)

    @property
    def split_fingerprint(self) -> str:
        return fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "strategy": self.strategy,
            "assignments": [asdict(item) for item in self.assignments],
        }
        if include_fingerprint:
            value["fingerprint"] = self.split_fingerprint
        return value

    def save(self, path: str | Path) -> None:
        atomic_json_dump(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> DatasetSplit:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            expected = raw.pop("fingerprint")
            result = cls(
                dataset=str(raw["dataset"]),
                seed=int(raw["seed"]),
                validation_fraction=float(raw["validation_fraction"]),
                strategy=str(raw["strategy"]),
                assignments=tuple(SplitAssignment(**item) for item in raw["assignments"]),
                schema_version=int(raw.get("schema_version", 1)),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatasetValidationError(f"invalid persisted split {source}: {exc}") from exc
        if result.split_fingerprint != expected:
            raise DatasetValidationError(f"persisted split fingerprint mismatch: {source}")
        validate_dataset_split(result)
        return result


def _stratification_key(record: SlideRecord, stratify_by_center: bool) -> tuple[object, ...]:
    return (record.label, record.center) if stratify_by_center else (record.label,)


def _validation_count(group_size: int, fraction: float) -> int:
    if group_size <= 1:
        return 0
    count = round(group_size * fraction)
    return min(group_size - 1, max(1, count))


def _slide_level_assignments(
    development: Sequence[SlideRecord],
    *,
    validation_fraction: float,
    seed: int,
    stratify_by_center: bool,
) -> list[SplitAssignment]:
    grouped: dict[tuple[object, ...], list[SlideRecord]] = {}
    for record in development:
        grouped.setdefault(_stratification_key(record, stratify_by_center), []).append(record)
    rng = np.random.default_rng(seed)
    validation: set[str] = set()
    for key in sorted(grouped, key=repr):
        records = sorted(grouped[key], key=lambda item: item.slide_id)
        order = rng.permutation(len(records))
        count = _validation_count(len(records), validation_fraction)
        validation.update(records[int(index)].slide_id for index in order[:count])
    return [
        SplitAssignment(
            slide_id=record.slide_id,
            label=record.label,
            split="val" if record.slide_id in validation else "train",
            patient_id=record.patient_id,
        )
        for record in development
    ]


def _patient_level_assignments(
    development: Sequence[SlideRecord],
    *,
    validation_fraction: float,
    seed: int,
    stratify_by_center: bool,
) -> list[SplitAssignment]:
    patients: dict[str, list[SlideRecord]] = {}
    for record in development:
        assert record.patient_id is not None
        patients.setdefault(record.patient_id, []).append(record)
    representatives: list[SlideRecord] = []
    for patient_id, records in patients.items():
        labels = {item.label for item in records}
        centers = {item.center for item in records}
        if len(labels) != 1:
            raise DatasetValidationError(
                f"patient {patient_id!r} has conflicting slide labels: {labels}"
            )
        if stratify_by_center and len(centers) != 1:
            raise DatasetValidationError(f"patient {patient_id!r} spans centers: {centers}")
        representatives.append(records[0])
    patient_assignments = _slide_level_assignments(
        representatives,
        validation_fraction=validation_fraction,
        seed=seed,
        stratify_by_center=stratify_by_center,
    )
    patient_split = {
        item.patient_id: item.split for item in patient_assignments if item.patient_id is not None
    }
    return [
        SplitAssignment(
            slide_id=record.slide_id,
            label=record.label,
            split=patient_split[cast(str, record.patient_id)],
            patient_id=record.patient_id,
        )
        for record in development
    ]


def create_dataset_split(
    records: Iterable[SlideRecord],
    *,
    dataset: str = "camelyon16",
    validation_fraction: float = 0.2,
    seed: int = 1,
    stratify_by_center: bool = True,
    prefer_patient_level: bool = True,
) -> DatasetSplit:
    """Preserve the official test cohort and split development records only."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    materialized = sorted(records, key=lambda item: item.slide_id)
    if not materialized:
        raise DatasetValidationError("cannot split an empty dataset")
    if len({item.slide_id for item in materialized}) != len(materialized):
        raise DatasetValidationError("slide IDs must be unique before splitting")
    development = [item for item in materialized if not item.is_official_test]
    official_test = [item for item in materialized if item.is_official_test]
    if not development or not official_test:
        raise DatasetValidationError("both development and official test cohorts are required")
    use_patients = prefer_patient_level and all(item.patient_id for item in development)
    if use_patients:
        assigned = _patient_level_assignments(
            development,
            validation_fraction=validation_fraction,
            seed=seed,
            stratify_by_center=stratify_by_center,
        )
        strategy = "patient_stratified"
    else:
        assigned = _slide_level_assignments(
            development,
            validation_fraction=validation_fraction,
            seed=seed,
            stratify_by_center=stratify_by_center,
        )
        strategy = "slide_stratified"
    assigned.extend(
        SplitAssignment(item.slide_id, item.label, "test", item.patient_id)
        for item in official_test
    )
    result = DatasetSplit(
        dataset=dataset,
        seed=seed,
        validation_fraction=validation_fraction,
        strategy=strategy,
        assignments=tuple(sorted(assigned, key=lambda item: item.slide_id)),
    )
    validate_dataset_split(result, expected_records=materialized)
    return result


def validate_dataset_split(
    split: DatasetSplit, *, expected_records: Sequence[SlideRecord] | None = None
) -> None:
    ids = [item.slide_id for item in split.assignments]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError("persisted split assigns a slide more than once")
    if not {"train", "val", "test"}.issubset({item.split for item in split.assignments}):
        raise DatasetValidationError(
            "persisted split must contain non-empty train, val, and test partitions"
        )
    patient_partitions: dict[str, set[str]] = {}
    for item in split.assignments:
        if item.patient_id:
            patient_partitions.setdefault(item.patient_id, set()).add(item.split)
    leaked = {key: value for key, value in patient_partitions.items() if len(value) > 1}
    if leaked:
        raise DatasetValidationError(f"patient leakage across split partitions: {leaked}")
    if expected_records is not None:
        expected = {item.slide_id: item for item in expected_records}
        if set(ids) != set(expected):
            raise DatasetValidationError(
                f"split slide set mismatch: missing={sorted(set(expected) - set(ids))[:8]}, "
                f"extra={sorted(set(ids) - set(expected))[:8]}"
            )
        for item in split.assignments:
            record = expected[item.slide_id]
            if item.label != record.label:
                raise DatasetValidationError(f"split label mismatch for {item.slide_id}")
            if record.is_official_test and item.split != "test":
                raise DatasetValidationError(
                    f"official test slide leaked into {item.split}: {item.slide_id}"
                )
            if not record.is_official_test and item.split == "test":
                raise DatasetValidationError(
                    f"development slide incorrectly assigned to test: {item.slide_id}"
                )


__all__ = [
    "DatasetSplit",
    "SplitAssignment",
    "create_dataset_split",
    "validate_dataset_split",
]
