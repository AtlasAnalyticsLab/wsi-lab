from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from wsi_lab.data.generic import GenericDatasetAdapter, ManifestDatasetAdapter
from wsi_lab.errors import DatasetValidationError

_FIELDS = (
    "slide_id",
    "image_path",
    "label",
    "label_name",
    "patient_id",
    "center",
    "official_cohort",
    "tissue_mask_path",
    "tumor_mask_path",
    "annotation_path",
)


def _row(slide_id: str = "slide_001", **updates: str) -> dict[str, str]:
    value = {
        "slide_id": slide_id,
        "image_path": f"images/{slide_id}.svs",
        "label": "0",
        "label_name": "negative",
        "patient_id": "patient_001",
        "center": "center_1",
        "official_cohort": "development",
        "tissue_mask_path": "",
        "tumor_mask_path": "",
        "annotation_path": "",
    }
    value.update(updates)
    return value


def _dataset(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    root = tmp_path / "custom-cohort"
    root.mkdir()
    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_manifest_reads_labeled_and_unlabeled_records(tmp_path: Path) -> None:
    root = _dataset(
        tmp_path,
        [
            _row("dev_slide", label="0", label_name="negative"),
            _row(
                "test_slide",
                label="1",
                label_name="positive",
                official_cohort="test",
            ),
            _row(
                "blind_slide",
                label="",
                label_name="",
                patient_id="",
                center="",
                official_cohort="test_unlabeled",
            ),
        ],
    )

    adapter = GenericDatasetAdapter(root)

    assert adapter.slide_ids() == ("dev_slide", "test_slide", "blind_slide")
    assert adapter.get_slide("dev_slide").label == 0
    assert adapter.get_slide("test_slide").label == 1
    blind = adapter.get_slide("blind_slide")
    assert blind.label is None
    assert blind.label_name == ""
    assert blind.patient_id is None
    assert blind.center is None
    assert blind.official_cohort == "test_unlabeled"
    assert blind.slide_path == (root / "images/blind_slide.svs").resolve()
    assert ManifestDatasetAdapter is GenericDatasetAdapter


def test_manifest_resolves_optional_sidecars_relative_to_dataset_root(tmp_path: Path) -> None:
    root = _dataset(
        tmp_path,
        [
            _row(
                tissue_mask_path="masks/tissue/slide_001.tif",
                tumor_mask_path="masks/tumor/slide_001.tif",
                annotation_path="annotations/slide_001.xml",
            )
        ],
    )

    record = GenericDatasetAdapter(root).get_slide("slide_001")

    assert record.tissue_mask_path == (root / "masks/tissue/slide_001.tif").resolve()
    assert record.tumor_mask_path == (root / "masks/tumor/slide_001.tif").resolve()
    assert record.annotation_path == (root / "annotations/slide_001.xml").resolve()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("image_path", "../outside.svs", "traversal"),
        ("image_path", "/private/outside.svs", "relative"),
        ("image_path", r"C:\\private\\outside.svs", "relative"),
        ("tissue_mask_path", "masks/../../outside.tif", "traversal"),
    ],
)
def test_manifest_rejects_unsafe_paths(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    root = _dataset(tmp_path, [_row(**{field: value})])

    with pytest.raises(DatasetValidationError, match=error):
        GenericDatasetAdapter(root)


def test_adapter_accepts_loader_resolved_config_paths_inside_root(tmp_path: Path) -> None:
    root = _dataset(tmp_path, [_row()])

    adapter = GenericDatasetAdapter(
        root,
        labels_path=(root / "metadata.csv").resolve(),
        embeddings_root=(root / "embeddings").resolve(),
    )

    assert adapter.labels_path == (root / "metadata.csv").resolve()
    assert adapter.embeddings_root == (root / "embeddings").resolve()


def test_adapter_rejects_config_paths_that_escape_root(tmp_path: Path) -> None:
    root = _dataset(tmp_path, [_row()])

    with pytest.raises(DatasetValidationError, match="escapes"):
        GenericDatasetAdapter(root, labels_path=tmp_path / "outside.csv")
    with pytest.raises(DatasetValidationError, match="traversal"):
        GenericDatasetAdapter(root, embeddings_root="../embeddings")


@pytest.mark.parametrize("slide_id", ["../slide", "nested/slide", r"nested\\slide", ".hidden", ""])
def test_manifest_rejects_unsafe_slide_ids(tmp_path: Path, slide_id: str) -> None:
    root = _dataset(tmp_path, [_row(slide_id)])

    with pytest.raises(DatasetValidationError, match="unsafe slide ID"):
        GenericDatasetAdapter(root)


def test_manifest_rejects_duplicate_slide_ids(tmp_path: Path) -> None:
    root = _dataset(tmp_path, [_row("duplicate"), _row("duplicate")])

    with pytest.raises(DatasetValidationError, match="duplicate slide ID"):
        GenericDatasetAdapter(root)


@pytest.mark.parametrize(
    ("label", "cohort"),
    [("", "development"), ("2", "development"), ("positive", "test")],
)
def test_manifest_enforces_binary_labels(tmp_path: Path, label: str, cohort: str) -> None:
    root = _dataset(tmp_path, [_row(label=label, official_cohort=cohort)])

    with pytest.raises(DatasetValidationError, match="label"):
        GenericDatasetAdapter(root)


def test_embedding_resolution_and_from_config(tmp_path: Path) -> None:
    root = _dataset(tmp_path, [_row("case-a")])
    dataset = SimpleNamespace(root=root, labels=(root / "metadata.csv").resolve())
    embeddings = SimpleNamespace(
        root=(root / "embeddings").resolve(),
        encoder="uni_v2",
        magnifications=(2.5, 20.0),
        directory_template="{encoder}/{magnification}x/patches",
        feature_key="/features/uni_v2/",
        coordinate_key="/coords/",
        passport_key="/passports/",
    )

    adapter = GenericDatasetAdapter.from_config(dataset, embeddings)
    location = adapter.embedding_location("case-a", 2.5)

    assert adapter.name == "generic"
    assert adapter.magnifications == (2.5, 20.0)
    assert location.path == (root / "embeddings/uni_v2/2.5x/patches/case-a.h5").resolve()
    assert location.feature_key == "features/uni_v2"
    assert location.coordinate_key == "coords"
    assert location.passport_key == "passports"
    with pytest.raises(KeyError):
        adapter.embedding_location("unknown", 2.5)
    with pytest.raises(KeyError):
        adapter.embedding_location("case-a", 10.0)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("embeddings_root", "PEAN"),
        ("embeddings_root", "bench_embeddings"),
        ("directory_template", "PEAN/{magnification}"),
        ("directory_template", "bench_embeddings/{magnification}"),
    ],
)
def test_excluded_embedding_sources_are_rejected(
    tmp_path: Path,
    argument: str,
    value: str,
) -> None:
    root = _dataset(tmp_path, [_row()])
    adapter: GenericDatasetAdapter | None = None

    if argument == "directory_template":
        adapter = GenericDatasetAdapter(root, directory_template=value)
        with pytest.raises(ValueError, match=r"PEAN|bench_embeddings"):
            adapter.embedding_location("slide_001", 2.5)
    else:
        with pytest.raises(ValueError, match=r"PEAN|bench_embeddings"):
            GenericDatasetAdapter(root, **{argument: value})


def test_manifest_requires_all_schema_columns(tmp_path: Path) -> None:
    root = tmp_path / "custom-cohort"
    root.mkdir()
    (root / "metadata.csv").write_text("slide_id,image_path\nslide,images/slide.svs\n")

    with pytest.raises(DatasetValidationError, match="missing columns"):
        GenericDatasetAdapter(root)
