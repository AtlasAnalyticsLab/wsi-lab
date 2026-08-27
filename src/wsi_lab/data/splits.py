"""Leakage-safe, persisted slide or patient splits."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from wsi_lab.data.base import SlideRecord
from wsi_lab.errors import DatasetValidationError
from wsi_lab.utils.atomic import atomic_json_dump
from wsi_lab.utils.fingerprints import fingerprint

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    slide_id: str
    label: int
    split: SplitName
    patient_id: str | None = None


@dataclass(frozen=True, slots=True)
class UnassignedSlide:
    slide_id: str
    reason: str
    patient_id: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    dataset: str
    seed: int
    validation_fraction: float
    strategy: str
    assignments: tuple[SplitAssignment, ...]
    test_fraction: float | None = None
    unassigned: tuple[UnassignedSlide, ...] = ()
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
        if self.schema_version >= 2 or self.unassigned:
            value["test_fraction"] = self.test_fraction
            value["unassigned"] = [asdict(item) for item in self.unassigned]
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
                test_fraction=(
                    float(raw["test_fraction"]) if raw.get("test_fraction") is not None else None
                ),
                unassigned=tuple(UnassignedSlide(**item) for item in raw.get("unassigned", ())),
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
            label=cast(int, record.label),
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
            label=cast(int, record.label),
            split=patient_split[cast(str, record.patient_id)],
            patient_id=record.patient_id,
        )
        for record in development
    ]


def _generated_group_assignments(
    development: Sequence[SlideRecord],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    stratify_by_center: bool,
    use_patients: bool,
) -> tuple[list[SplitAssignment], str]:
    grouped: dict[str, list[SlideRecord]] = {}
    for record in development:
        group_id = cast(str, record.patient_id) if use_patients else record.slide_id
        grouped.setdefault(group_id, []).append(record)

    units: list[tuple[str, list[SlideRecord], int, str | None]] = []
    for group_id, records in sorted(grouped.items()):
        centers = {item.center for item in records}
        if stratify_by_center and len(centers) > 1:
            raise DatasetValidationError(f"group {group_id!r} spans centers: {centers}")
        # A CAMELYON17 patient may legitimately have both positive and negative
        # lymph nodes. Group-level stratification uses the maximum slide label
        # while every slide retains its own label in the assignment manifest.
        units.append(
            (group_id, records, max(cast(int, item.label) for item in records), records[0].center)
        )

    def keys(include_center: bool) -> dict[tuple[object, ...], list[int]]:
        result: dict[tuple[object, ...], list[int]] = {}
        for index, (_, _, label, center) in enumerate(units):
            key = (label, center) if include_center else (label,)
            result.setdefault(key, []).append(index)
        return result

    strata = keys(stratify_by_center)
    center_fallback = stratify_by_center and any(len(indices) < 3 for indices in strata.values())
    if center_fallback:
        strata = keys(False)
    if any(len(indices) < 3 for indices in strata.values()):
        raise DatasetValidationError(
            "generated train/val/test split requires at least three patient/slide groups "
            "in every label stratum"
        )

    rng = np.random.default_rng(seed)
    partition_by_unit: dict[int, SplitName] = {}
    for key in sorted(strata, key=repr):
        indices = np.asarray(strata[key], dtype=np.int64)
        order = indices[rng.permutation(len(indices))]
        test_count = max(1, round(len(order) * test_fraction))
        validation_count = max(1, round(len(order) * validation_fraction))
        while test_count + validation_count >= len(order):
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                raise DatasetValidationError(f"cannot split stratum {key!r} without empty train")
        for index in order[:test_count]:
            partition_by_unit[int(index)] = "test"
        for index in order[test_count : test_count + validation_count]:
            partition_by_unit[int(index)] = "val"
        for index in order[test_count + validation_count :]:
            partition_by_unit[int(index)] = "train"

    assignments = [
        SplitAssignment(
            slide_id=record.slide_id,
            label=cast(int, record.label),
            split=partition_by_unit[index],
            patient_id=record.patient_id,
        )
        for index, (_, records, _, _) in enumerate(units)
        for record in records
    ]
    level = "patient" if use_patients else "slide"
    stratification = "label" if center_fallback or not stratify_by_center else "label_center"
    return assignments, f"generated_{level}_{stratification}"


def create_dataset_split(
    records: Iterable[SlideRecord],
    *,
    dataset: str = "camelyon16",
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 1,
    stratify_by_center: bool = True,
    prefer_patient_level: bool = True,
) -> DatasetSplit:
    """Preserve a labeled native test cohort or generate a leakage-safe test split."""

    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation and test fractions must be strictly between zero and one")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than one")
    materialized = sorted(records, key=lambda item: item.slide_id)
    if not materialized:
        raise DatasetValidationError("cannot split an empty dataset")
    if len({item.slide_id for item in materialized}) != len(materialized):
        raise DatasetValidationError("slide IDs must be unique before splitting")
    labeled = [item for item in materialized if item.is_labeled]
    unassigned = tuple(
        UnassignedSlide(item.slide_id, "missing_label", item.patient_id)
        for item in materialized
        if not item.is_labeled
    )
    development = [item for item in labeled if not item.is_official_test]
    official_test = [item for item in labeled if item.is_official_test]
    if not development:
        raise DatasetValidationError("dataset has no labeled development records")
    use_patients = prefer_patient_level and all(item.patient_id for item in development)
    if official_test:
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
            SplitAssignment(item.slide_id, cast(int, item.label), "test", item.patient_id)
            for item in official_test
        )
    else:
        assigned, strategy = _generated_group_assignments(
            development,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            stratify_by_center=stratify_by_center,
            use_patients=use_patients,
        )
    result = DatasetSplit(
        dataset=dataset,
        seed=seed,
        validation_fraction=validation_fraction,
        strategy=strategy,
        assignments=tuple(sorted(assigned, key=lambda item: item.slide_id)),
        test_fraction=None if official_test else test_fraction,
        unassigned=tuple(sorted(unassigned, key=lambda item: item.slide_id)),
        schema_version=2 if unassigned or not official_test else 1,
    )
    validate_dataset_split(result, expected_records=materialized)
    return result


def validate_dataset_split(
    split: DatasetSplit, *, expected_records: Sequence[SlideRecord] | None = None
) -> None:
    ids = [item.slide_id for item in split.assignments]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError("persisted split assigns a slide more than once")
    unassigned_ids = [item.slide_id for item in split.unassigned]
    if len(unassigned_ids) != len(set(unassigned_ids)):
        raise DatasetValidationError("persisted split lists an unassigned slide more than once")
    overlap = set(ids).intersection(unassigned_ids)
    if overlap:
        raise DatasetValidationError(
            f"slides are both assigned and unassigned: {sorted(overlap)[:8]}"
        )
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
        represented = set(ids).union(unassigned_ids)
        if represented != set(expected):
            raise DatasetValidationError(
                f"split slide set mismatch: missing={sorted(set(expected) - represented)[:8]}, "
                f"extra={sorted(represented - set(expected))[:8]}"
            )
        for item in split.assignments:
            record = expected[item.slide_id]
            if record.label is None:
                raise DatasetValidationError(f"unlabeled slide was assigned: {item.slide_id}")
            if item.label != record.label:
                raise DatasetValidationError(f"split label mismatch for {item.slide_id}")
            if record.is_official_test and item.split != "test":
                raise DatasetValidationError(
                    f"official test slide leaked into {item.split}: {item.slide_id}"
                )
            if (
                not record.is_official_test
                and item.split == "test"
                and not split.strategy.startswith("generated_")
                and not (
                    split.dataset == "camelyon16_17"
                    and split.strategy == "combined_corresponding"
                    and record.metadata.get("source_dataset") == "camelyon17"
                )
            ):
                raise DatasetValidationError(
                    f"development slide incorrectly assigned to test: {item.slide_id}"
                )
        for unassigned_item in split.unassigned:
            record = expected[unassigned_item.slide_id]
            if record.is_labeled:
                raise DatasetValidationError(
                    f"labeled slide is unexpectedly unassigned: {unassigned_item.slide_id}"
                )


__all__ = [
    "DatasetSplit",
    "SplitAssignment",
    "UnassignedSlide",
    "create_dataset_split",
    "validate_dataset_split",
]
