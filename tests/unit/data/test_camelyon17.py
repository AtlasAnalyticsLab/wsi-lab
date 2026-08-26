from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from wsi_lab.constants import BELIEF_MAGNIFICATIONS
from wsi_lab.data.camelyon17 import Camelyon17Adapter
from wsi_lab.errors import DatasetValidationError


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "CAMELYON17"
    for directory in ("images", "masks", "annotations"):
        (root / directory).mkdir(parents=True)
    for filename in (
        "patient_000_node_0.tif",
        "patient_000_node_1.tif",
        "patient_100_node_0.tif",
    ):
        (root / "images" / filename).touch()
    (root / "masks" / "patient_000_node_1_mask.tif").touch()
    (root / "annotations" / "patient_000_node_1.xml").touch()
    (root / "stages.csv").write_text(
        "patient,stage,center\n"
        "patient_000.zip,pN1,0\n"
        "patient_000_node_0.tif,negative,0\n"
        "patient_000_node_1.tif,macro,0\n",
        encoding="utf-8",
    )
    return root


def test_native_discovery_preserves_labeled_and_unlabeled_slides(tmp_path: Path) -> None:
    adapter = Camelyon17Adapter(_dataset(tmp_path))

    assert adapter.slide_ids() == (
        "patient_000_node_0",
        "patient_000_node_1",
        "patient_100_node_0",
    )

    negative = adapter.get_slide("patient_000_node_0")
    assert negative.label == 0
    assert negative.label_name == "negative"
    assert negative.patient_id == "patient_000"
    assert negative.center == "0"
    assert negative.official_cohort == "development"

    positive = adapter.get_slide("patient_000_node_1")
    assert positive.label == 1
    assert positive.label_name == "macro"
    assert positive.tumor_class == "macro"
    assert positive.tumor_mask_path == tmp_path / "CAMELYON17/masks/patient_000_node_1_mask.tif"
    assert positive.annotation_path == tmp_path / "CAMELYON17/annotations/patient_000_node_1.xml"

    unlabeled = adapter.get_slide("patient_100_node_0")
    assert unlabeled.label is None
    assert unlabeled.label_name == "unlabeled"
    assert unlabeled.patient_id == "patient_100"
    assert unlabeled.center is None
    assert unlabeled.official_cohort == "test_unlabeled"


def test_embedding_location_uses_native_scale_directories(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    adapter = Camelyon17Adapter(root)

    location = adapter.embedding_location("patient_000_node_1", 2.5)

    assert (
        location.path
        == (root / "Embeddings/UNI_v2_2.5x_256/patches/patient_000_node_1.h5").resolve()
    )
    assert location.feature_key == "features/uni_v2"
    assert location.coordinate_key == "coords"
    assert location.passport_key == "passports"


def test_from_config_resolves_configured_paths(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    dataset = SimpleNamespace(
        root=root,
        slides_dir=root / "images",
        labels=root / "stages.csv",
        tissue_masks_dir=root / "tissue",
        tumor_masks_dir=root / "masks",
        annotations_dir=root / "annotations",
    )
    embeddings = SimpleNamespace(
        root=root / "custom_embeddings",
        encoder="uni_v2",
        magnifications=BELIEF_MAGNIFICATIONS,
        directory_template="scale_{magnification}/{slide_id}",
        feature_key="features/custom",
        coordinate_key="custom_coords",
        passport_key=None,
    )

    adapter = Camelyon17Adapter.from_config(dataset, embeddings)
    location = adapter.embedding_location("patient_000_node_0", 20.0)

    assert adapter.labels_path == (root / "stages.csv").resolve()
    assert (
        location.path
        == (root / "custom_embeddings/scale_20/patient_000_node_0/patient_000_node_0.h5").resolve()
    )
    assert location.feature_key == "features/custom"
    assert location.coordinate_key == "custom_coords"
    assert location.passport_key is None


@pytest.mark.parametrize(
    ("stage", "expected"),
    [("negative", 0), ("itc", 1), ("micro", 1), ("macro", 1)],
)
def test_stage_mapping(tmp_path: Path, stage: str, expected: int) -> None:
    root = _dataset(tmp_path)
    (root / "stages.csv").write_text(
        f"patient,stage,center\n"
        f"patient_000_node_0.tif,{stage},2\n"
        "patient_000_node_1.tif,negative,2\n",
        encoding="utf-8",
    )

    record = Camelyon17Adapter(root).get_slide("patient_000_node_0")

    assert record.label == expected
    assert record.label_name == stage
    assert record.center == "2"


def test_missing_development_stage_row_is_not_reclassified_as_blind_test(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path)
    (root / "stages.csv").write_text(
        "patient,stage,center\npatient_000_node_0.tif,negative,0\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match=r"development slide.*has no stage row"):
        Camelyon17Adapter(root)


def test_rejects_malformed_image_id(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    (root / "images" / "patient-001-node-0.tif").touch()

    with pytest.raises(DatasetValidationError, match="invalid CAMELYON17 slide filename"):
        Camelyon17Adapter(root)


def test_rejects_malformed_stage_id(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    (root / "stages.csv").write_text(
        "patient,stage,center\npatient-000-node-0.tif,negative,0\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="invalid CAMELYON17 stage image"):
        Camelyon17Adapter(root)


def test_rejects_duplicate_stage_rows(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    (root / "stages.csv").write_text(
        "patient,stage,center\npatient_000_node_0.tif,negative,0\npatient_000_node_0.tif,macro,0\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="duplicate CAMELYON17 stage row"):
        Camelyon17Adapter(root)


def test_rejects_stage_row_without_image(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    (root / "stages.csv").write_text(
        "patient,stage,center\npatient_001_node_0.tif,negative,0\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="no matching image"):
        Camelyon17Adapter(root)


@pytest.mark.parametrize("excluded", ["PEAN", "bench_embeddings"])
def test_excluded_embedding_roots_are_rejected(tmp_path: Path, excluded: str) -> None:
    root = _dataset(tmp_path)

    with pytest.raises(ValueError, match=r"PEAN|bench_embeddings"):
        Camelyon17Adapter(root, embeddings_root=tmp_path / excluded)


def test_embedding_template_rejects_escape(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    adapter = Camelyon17Adapter(root, directory_template="../outside/{magnification}")

    with pytest.raises(ValueError, match="traversal"):
        adapter.embedding_location("patient_000_node_0", 2.5)
