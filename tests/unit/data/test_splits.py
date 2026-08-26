from __future__ import annotations

from pathlib import Path

from wsi_lab.data.base import SlideRecord
from wsi_lab.data.splits import DatasetSplit, create_dataset_split


def _record(
    slide_id: str,
    label: int | None,
    *,
    patient_id: str | None = None,
    cohort: str = "development",
    center: str | None = None,
) -> SlideRecord:
    return SlideRecord(
        slide_id=slide_id,
        slide_path=Path("images") / f"{slide_id}.tif",
        label=label,
        label_name="unlabeled" if label is None else str(label),
        official_cohort=cohort,
        patient_id=patient_id,
        center=center,
    )


def test_official_test_policy_preserves_legacy_schema_and_membership() -> None:
    records = [
        *(_record(f"normal_{index:03d}", 0, center="0") for index in range(1, 6)),
        *(_record(f"tumor_{index:03d}", 1, center="0") for index in range(1, 6)),
        _record("test_001", 0, cohort="test"),
        _record("test_002", 1, cohort="test"),
    ]

    split = create_dataset_split(records, dataset="camelyon16", seed=1)

    assert split.schema_version == 1
    assert split.strategy == "slide_stratified"
    assert split.ids("test") == ("test_001", "test_002")
    assert split.unassigned == ()
    assert "unassigned" not in split.to_dict()


def test_dataset_without_official_test_gets_patient_grouped_three_way_split() -> None:
    records = []
    for label in (0, 1):
        for patient_index in range(10):
            patient_id = f"patient_{label}_{patient_index}"
            records.extend(
                (
                    _record(
                        f"{patient_id}_a",
                        label,
                        patient_id=patient_id,
                        center="site-a",
                    ),
                    _record(
                        f"{patient_id}_b",
                        label,
                        patient_id=patient_id,
                        center="site-a",
                    ),
                )
            )

    split = create_dataset_split(
        records,
        dataset="tcga_nsclc",
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=4,
    )

    assert split.strategy == "generated_patient_label_center"
    assert all(split.ids(name) for name in ("train", "val", "test"))
    patient_partitions: dict[str, set[str]] = {}
    for assignment in split.assignments:
        assert assignment.patient_id is not None
        patient_partitions.setdefault(assignment.patient_id, set()).add(assignment.split)
    assert all(len(partitions) == 1 for partitions in patient_partitions.values())


def test_mixed_slide_labels_within_patient_remain_in_one_generated_partition() -> None:
    records = []
    for patient_index in range(8):
        patient_id = f"patient_{patient_index:03d}"
        records.extend(
            (
                _record(f"{patient_id}_negative", 0, patient_id=patient_id),
                _record(f"{patient_id}_positive", 1, patient_id=patient_id),
            )
        )

    split = create_dataset_split(records, dataset="camelyon17", stratify_by_center=False)

    for patient_id in {record.patient_id for record in records}:
        partitions = {item.split for item in split.assignments if item.patient_id == patient_id}
        assert len(partitions) == 1


def test_unlabeled_official_slides_are_persisted_as_unassigned(tmp_path: Path) -> None:
    records = [
        *(
            _record(f"patient_{index:03d}_node_0", index % 2, patient_id=f"patient_{index:03d}")
            for index in range(12)
        ),
        _record(
            "patient_100_node_0",
            None,
            patient_id="patient_100",
            cohort="test_unlabeled",
        ),
    ]
    split = create_dataset_split(records, dataset="camelyon17", stratify_by_center=False)
    path = tmp_path / "camelyon17_split.json"
    split.save(path)

    loaded = DatasetSplit.load(path)

    assert loaded.schema_version == 2
    assert loaded.unassigned[0].slide_id == "patient_100_node_0"
    assert loaded.unassigned[0].reason == "missing_label"
    assert "patient_100_node_0" not in {
        slide_id for name in ("train", "val", "test") for slide_id in loaded.ids(name)
    }
