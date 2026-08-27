from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from wsi_lab.constants import BELIEF_MAGNIFICATIONS
from wsi_lab.data.base import EmbeddingLocation, SlideRecord
from wsi_lab.data.camelyon16_17 import CombinedCamelyonAdapter
from wsi_lab.data.factory import create_dataset_adapter
from wsi_lab.data.splits import DatasetSplit, validate_dataset_split
from wsi_lab.errors import DatasetValidationError


def _write_camelyon16(root: Path) -> None:
    images = root / "images"
    labels = root / "evaluation" / "reference.csv"
    images.mkdir(parents=True)
    labels.parent.mkdir(parents=True)

    rows = ["image,type,class,center"]
    for label_name, tumor_class in (("normal", "negative"), ("tumor", "macro")):
        for index in range(1, 7):
            filename = f"{label_name}_{index:03d}.tif"
            (images / filename).touch()
            rows.append(f"{filename},{label_name},{tumor_class},c16")
    for index, (label_name, tumor_class) in enumerate(
        (("normal", "negative"), ("tumor", "micro")),
        start=1,
    ):
        filename = f"test_{index:03d}.tif"
        (images / filename).touch()
        rows.append(f"{filename},{label_name},{tumor_class},c16-test")
    labels.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_camelyon17(root: Path) -> None:
    images = root / "images"
    images.mkdir(parents=True)
    rows = ["patient,stage,center"]

    # Six negative patients and six mixed negative/positive patients provide
    # enough patient groups in both strata for deterministic three-way splits.
    for patient_number in range(12):
        positive_stage = ("itc", "micro", "macro")[patient_number % 3]
        for node_number in range(2):
            filename = f"patient_{patient_number:03d}_node_{node_number}.tif"
            (images / filename).touch()
            stage = positive_stage if patient_number >= 6 and node_number == 1 else "negative"
            rows.append(f"{filename},{stage},c17")

    # Native CAMELYON17 patients 100+ have no slide-level labels and must not
    # enter any learning or evaluation partition.
    for node_number in range(2):
        (images / f"patient_100_node_{node_number}.tif").touch()
    (root / "stages.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def combined_root(tmp_path: Path) -> Path:
    root = tmp_path / "CAMELYON"
    _write_camelyon16(root / "CAMELYON16")
    _write_camelyon17(root / "CAMELYON17")
    return root


def _split_kwargs(seed: int = 7) -> dict[str, object]:
    return {
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "seed": seed,
        "stratify_by_center": True,
        "prefer_patient_level": True,
    }


def test_combined_adapter_preserves_provenance_binary_labels_and_artifact_dispatch(
    combined_root: Path,
) -> None:
    adapter = CombinedCamelyonAdapter(combined_root)

    assert adapter.name == "camelyon16_17"
    assert adapter.root == combined_root.resolve()
    assert adapter.slide_ids() == tuple(sorted(adapter.slide_ids()))
    assert len(adapter.slide_ids()) == 40
    assert adapter.metadata_paths == (
        (combined_root / "CAMELYON16/evaluation/reference.csv").resolve(),
        (combined_root / "CAMELYON17/stages.csv").resolve(),
    )

    c16_normal = adapter.get_slide("normal_001")
    c16_tumor = adapter.get_slide("tumor_001")
    c17_negative = adapter.get_slide("patient_000_node_0")
    assert c16_normal.label == 0
    assert c16_tumor.label == 1
    assert c17_negative.label == 0
    assert c16_normal.metadata["source_dataset"] == "camelyon16"
    assert c16_tumor.metadata["reference_image"] == "tumor_001.tif"
    assert c17_negative.metadata["source_dataset"] == "camelyon17"
    assert c17_negative.metadata["patient_number"] == 0

    for patient_number, expected_stage in ((6, "itc"), (7, "micro"), (8, "macro")):
        positive = adapter.get_slide(f"patient_{patient_number:03d}_node_1")
        assert positive.label == 1
        assert positive.label_name == expected_stage
        assert positive.tumor_class == expected_stage

    c16_embedding = adapter.embedding_location("normal_001", 2.5)
    c17_embedding = adapter.embedding_location("patient_006_node_1", 20.0)
    assert (
        c16_embedding.path
        == (combined_root / "CAMELYON16/Embeddings/UNI_v2_2.5x_256/patches/normal_001.h5").resolve()
    )
    assert (
        c17_embedding.path
        == (
            combined_root / "CAMELYON17/Embeddings/UNI_v2_20x_256/patches/patient_006_node_1.h5"
        ).resolve()
    )
    assert c16_embedding.feature_key == c17_embedding.feature_key == "features/uni_v2"


def test_combined_split_is_exact_union_of_native_corresponding_splits(
    combined_root: Path,
) -> None:
    adapter = CombinedCamelyonAdapter(combined_root)
    kwargs = _split_kwargs()
    combined = adapter.create_split(**kwargs)
    component_splits = tuple(source.create_split(**kwargs) for source in adapter.source_adapters)

    assert combined.dataset == adapter.name
    assert combined.strategy == "combined_corresponding"
    assert combined.schema_version == 2
    assert combined.test_fraction == 0.2
    for partition in ("train", "val", "test"):
        expected = tuple(
            sorted(slide_id for split in component_splits for slide_id in split.ids(partition))
        )
        assert combined.ids(partition) == expected
    assert combined.unassigned == tuple(
        sorted(
            (item for split in component_splits for item in split.unassigned),
            key=lambda item: item.slide_id,
        )
    )

    # The small native fixtures have 10/2/2 C16 assignments and 16/4/4
    # labeled C17 assignments, respectively.
    assert {partition: len(combined.ids(partition)) for partition in ("train", "val", "test")} == {
        "train": 26,
        "val": 6,
        "test": 6,
    }


def test_combined_split_keeps_c17_patients_together_and_blind_slides_unassigned(
    combined_root: Path,
) -> None:
    adapter = CombinedCamelyonAdapter(combined_root)
    split = adapter.create_split(**_split_kwargs())

    for patient_number in range(12):
        patient_id = f"patient_{patient_number:03d}"
        partitions = {
            assignment.split
            for assignment in split.assignments
            if assignment.patient_id == patient_id
        }
        assert len(partitions) == 1

    blind_ids = {"patient_100_node_0", "patient_100_node_1"}
    assert {item.slide_id for item in split.unassigned} == blind_ids
    assert all(item.reason == "missing_label" for item in split.unassigned)
    assert blind_ids.isdisjoint(assignment.slide_id for assignment in split.assignments)

    source16, source17 = adapter.source_adapters
    c16_development = {
        record.slide_id for record in source16.iter_slides() if not record.is_official_test
    }
    assert c16_development.isdisjoint(split.ids("test"))
    c17_generated_test = set(source17.slide_ids()).intersection(split.ids("test"))
    assert c17_generated_test
    assert all(adapter.get_slide(slide_id).is_labeled for slide_id in c17_generated_test)


def test_combined_split_is_deterministic_and_survives_persistence(
    combined_root: Path,
    tmp_path: Path,
) -> None:
    adapter = CombinedCamelyonAdapter(combined_root)
    first = adapter.create_split(**_split_kwargs(seed=19))
    second = adapter.create_split(**_split_kwargs(seed=19))
    different_seed = adapter.create_split(**_split_kwargs(seed=20))

    assert first == second
    assert first.split_fingerprint == second.split_fingerprint
    assert first.assignments != different_seed.assignments

    path = tmp_path / "camelyon16_17_split.json"
    first.save(path)
    loaded = DatasetSplit.load(path)

    assert loaded == first
    validate_dataset_split(loaded, expected_records=tuple(adapter.iter_slides()))


def test_combined_validation_does_not_allow_c16_development_into_test(
    combined_root: Path,
) -> None:
    adapter = CombinedCamelyonAdapter(combined_root)
    split = adapter.create_split(**_split_kwargs())
    tampered = replace(
        split,
        assignments=tuple(
            replace(item, split="test") if item.slide_id == "normal_001" else item
            for item in split.assignments
        ),
    )

    with pytest.raises(DatasetValidationError, match="development slide incorrectly assigned"):
        validate_dataset_split(tampered, expected_records=tuple(adapter.iter_slides()))


def test_factory_builds_combined_adapter_with_resolved_shared_layout(
    combined_root: Path,
) -> None:
    dataset = SimpleNamespace(
        name="camelyon16_17",
        root=combined_root.resolve(),
        camelyon16_root=(combined_root / "CAMELYON16").resolve(),
        camelyon17_root=(combined_root / "CAMELYON17").resolve(),
        slides_dir=(combined_root / "images").resolve(),
        labels=(combined_root / "CAMELYON16/evaluation/reference.csv").resolve(),
        camelyon17_labels=(combined_root / "CAMELYON17/stages.csv").resolve(),
        tissue_masks_dir=(combined_root / "background_tissue").resolve(),
        tumor_masks_dir=(combined_root / "masks").resolve(),
        annotations_dir=(combined_root / "annotations").resolve(),
    )
    embeddings = SimpleNamespace(
        root=(combined_root / "Embeddings").resolve(),
        encoder="uni_v2",
        magnifications=BELIEF_MAGNIFICATIONS,
        directory_template="UNI_v2_{magnification}x_256/patches",
        feature_key="features/uni_v2",
        coordinate_key="coords",
        passport_key="passports",
    )

    adapter = create_dataset_adapter(dataset, embeddings)

    assert isinstance(adapter, CombinedCamelyonAdapter)
    assert adapter.get_slide("test_002").metadata["source_dataset"] == "camelyon16"
    assert adapter.get_slide("patient_100_node_0").label is None


class _CollisionAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.magnifications = BELIEF_MAGNIFICATIONS
        self._record = SlideRecord(
            slide_id="shared_slide",
            slide_path=Path(name) / "shared_slide.tif",
            label=0,
            label_name="negative",
            official_cohort="development",
        )

    def iter_slides(self) -> Iterator[SlideRecord]:
        return iter((self._record,))

    def embedding_location(self, slide_id: str, magnification: float) -> EmbeddingLocation:
        return EmbeddingLocation(
            slide_id=slide_id,
            magnification=magnification,
            path=Path(self.name) / f"{slide_id}.h5",
            encoder="uni_v2",
            feature_key="features/uni_v2",
        )


def test_combined_adapter_rejects_cross_source_slide_id_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "wsi_lab.data.camelyon16_17.Camelyon16Adapter",
        lambda *args, **kwargs: _CollisionAdapter("camelyon16"),
    )
    monkeypatch.setattr(
        "wsi_lab.data.camelyon16_17.Camelyon17Adapter",
        lambda *args, **kwargs: _CollisionAdapter("camelyon17"),
    )

    with pytest.raises(DatasetValidationError, match="duplicate slide ID across CAMELYON"):
        CombinedCamelyonAdapter(tmp_path)
