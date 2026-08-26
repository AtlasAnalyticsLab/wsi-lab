from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from wsi_lab.data.tcga_nsclc import TCGANsclcAdapter
from wsi_lab.errors import DatasetValidationError

_LUAD_ONE = "TCGA-55-7576-01Z-00-DX1.3a6142e9-c90a-4938-a1d1-061c2ac261f0"
_LUAD_TWO = "TCGA-64-5775-01Z-00-DX1.9A338C1A-CDB6-41AF-B6D5-EDE608B5E3E1"
_LUSC_ONE = "TCGA-22-4604-01Z-00-DX1.C398D192-8564-42AB-89B8-461F0830D8A1"


def _native_dataset(
    tmp_path: Path,
    *,
    luad_ids: tuple[str, ...] = (_LUAD_ONE,),
    lusc_ids: tuple[str, ...] = (_LUSC_ONE,),
    luad_dir: str = "TCGA-LUAD",
    lusc_dir: str = "TCGA-LUSC",
) -> Path:
    root = tmp_path / "TCGA-NSCLC"
    for directory, slide_ids in ((luad_dir, luad_ids), (lusc_dir, lusc_ids)):
        images = root / directory / "images"
        images.mkdir(parents=True)
        for slide_id in slide_ids:
            (images / f"{slide_id}.svs").touch()
    return root


def test_native_cohorts_are_normalized_and_sorted(tmp_path: Path) -> None:
    root = _native_dataset(
        tmp_path,
        luad_ids=(_LUAD_TWO, _LUAD_ONE),
    )
    (root / "TCGA-LUAD" / "images" / "README.txt").touch()

    adapter = TCGANsclcAdapter(root)

    assert adapter.name == "tcga_nsclc"
    assert adapter.slide_ids() == tuple(sorted((_LUAD_ONE, _LUAD_TWO, _LUSC_ONE)))

    luad = adapter.get_slide(_LUAD_ONE)
    assert luad.slide_id == _LUAD_ONE
    assert luad.slide_path == root / "TCGA-LUAD" / "images" / f"{_LUAD_ONE}.svs"
    assert luad.label == 0
    assert luad.label_name == "luad"
    assert luad.tumor_class == "luad"
    assert luad.patient_id == "TCGA-55-7576"
    assert luad.center == "55"
    assert luad.official_cohort == "development"
    assert luad.metadata == {"source_cohort": "TCGA-LUAD"}
    assert luad.tissue_mask_path is None
    assert luad.tumor_mask_path is None
    assert luad.annotation_path is None

    lusc = adapter.get_slide(_LUSC_ONE)
    assert lusc.label == 1
    assert lusc.label_name == "lusc"
    assert lusc.patient_id == "TCGA-22-4604"
    assert lusc.center == "22"
    assert lusc.metadata == {"source_cohort": "TCGA-LUSC"}


def test_embedding_location_uses_the_owning_cohort_root(tmp_path: Path) -> None:
    root = _native_dataset(tmp_path)
    adapter = TCGANsclcAdapter(root)

    luad = adapter.embedding_location(_LUAD_ONE, 2.5)
    lusc = adapter.embedding_location(_LUSC_ONE, 20.0)

    assert luad.path == (
        root / "TCGA-LUAD" / "Embeddings" / "UNI_v2_2.5x_256" / "patches" / f"{_LUAD_ONE}.h5"
    )
    assert lusc.path == (
        root / "TCGA-LUSC" / "Embeddings" / "UNI_v2_20x_256" / "patches" / f"{_LUSC_ONE}.h5"
    )
    assert luad.feature_key == "features/uni_v2"
    assert luad.coordinate_key == "coords"
    assert luad.passport_key == "passports"


def test_from_config_uses_native_cohort_and_embedding_fields(tmp_path: Path) -> None:
    root = _native_dataset(tmp_path, luad_dir="LUAD-native", lusc_dir="LUSC-native")
    dataset = SimpleNamespace(
        root=root,
        luad_dir=Path("LUAD-native"),
        lusc_dir=Path("LUSC-native"),
    )
    embeddings = SimpleNamespace(
        root=Path("Features"),
        encoder="custom_encoder",
        magnifications=(2.5, 5.0),
        directory_template="{encoder}/{magnification}/patches",
        feature_key="features/custom",
        coordinate_key="locations",
        passport_key=None,
    )

    adapter = TCGANsclcAdapter.from_config(dataset, embeddings)
    location = adapter.embedding_location(_LUAD_ONE, 5.0)

    assert adapter.magnifications == (2.5, 5.0)
    assert location.path == (
        root / "LUAD-native" / "Features" / "custom_encoder" / "5" / "patches" / f"{_LUAD_ONE}.h5"
    )
    assert location.feature_key == "features/custom"
    assert location.coordinate_key == "locations"
    assert location.passport_key is None


@pytest.mark.parametrize(
    "slide_id",
    [
        "not-a-tcga-barcode",
        "TCGA-A-1234-01Z-00-DX1.uuid",
        "TCGA-AB-123-01Z-00-DX1.uuid",
        "TCGA-AB-1234",
    ],
)
def test_invalid_tcga_slide_ids_fail_discovery(tmp_path: Path, slide_id: str) -> None:
    root = _native_dataset(tmp_path, luad_ids=(slide_id,))

    with pytest.raises(DatasetValidationError, match="invalid TCGA slide ID"):
        TCGANsclcAdapter(root)


def test_duplicate_slide_id_across_cohorts_is_rejected(tmp_path: Path) -> None:
    root = _native_dataset(tmp_path, luad_ids=(_LUAD_ONE,), lusc_ids=(_LUAD_ONE,))

    with pytest.raises(DatasetValidationError, match="duplicate TCGA slide ID"):
        TCGANsclcAdapter(root)


def test_patient_with_conflicting_subtype_is_rejected(tmp_path: Path) -> None:
    luad = "TCGA-AB-1234-01Z-00-DX1.first"
    lusc = "TCGA-AB-1234-01Z-00-DX2.second"
    root = _native_dataset(tmp_path, luad_ids=(luad,), lusc_ids=(lusc,))

    with pytest.raises(DatasetValidationError, match="conflicting classes"):
        TCGANsclcAdapter(root)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("embeddings_root", "bench_embeddings", "bench_embeddings"),
        ("embeddings_root", "../Embeddings", "traversal"),
        ("luad_dir", "../TCGA-LUAD", "traversal"),
    ],
)
def test_unsafe_native_paths_are_rejected(
    tmp_path: Path, keyword: str, value: str, message: str
) -> None:
    root = _native_dataset(tmp_path)

    with pytest.raises(ValueError, match=message):
        TCGANsclcAdapter(root, **{keyword: value})


@pytest.mark.parametrize(
    "template",
    ["../outside/{magnification}", "bench_embeddings/{magnification}"],
)
def test_unsafe_embedding_template_is_rejected_at_resolution(tmp_path: Path, template: str) -> None:
    adapter = TCGANsclcAdapter(
        _native_dataset(tmp_path),
        directory_template=template,
    )

    with pytest.raises(ValueError, match=r"traversal|bench_embeddings"):
        adapter.embedding_location(_LUAD_ONE, 2.5)


def test_missing_or_empty_image_cohort_is_reported(tmp_path: Path) -> None:
    root = _native_dataset(tmp_path)
    (root / "TCGA-LUSC" / "images" / f"{_LUSC_ONE}.svs").unlink()

    with pytest.raises(DatasetValidationError, match=r"LUSC.*no \.svs files"):
        TCGANsclcAdapter(root)
