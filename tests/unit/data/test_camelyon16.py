from __future__ import annotations

from pathlib import Path

import pytest

from wsi_lab.data.camelyon16 import Camelyon16Adapter


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "CAMELYON16"
    (root / "evaluation").mkdir(parents=True)
    (root / "evaluation" / "reference.csv").write_text(
        "image,type,class,center\n"
        "normal_001.tif,normal,negative,c0\n"
        "tumor_001.tif,tumor,macro,c1\n"
        "test_001.tif,tumor,micro,c2\n",
        encoding="utf-8",
    )
    return root


def test_reference_table_drives_labels_and_official_cohort(tmp_path: Path) -> None:
    adapter = Camelyon16Adapter(_dataset(tmp_path))

    assert adapter.get_slide("normal_001").label == 0
    assert adapter.get_slide("tumor_001").label == 1
    test = adapter.get_slide("test_001")
    assert test.label == 1
    assert test.is_official_test
    assert adapter.slide_ids() == ("normal_001", "tumor_001", "test_001")


@pytest.mark.parametrize("excluded", ["PEAN", "bench_embeddings"])
def test_excluded_embedding_roots_are_rejected(tmp_path: Path, excluded: str) -> None:
    root = _dataset(tmp_path)
    with pytest.raises(ValueError, match=r"PEAN|bench_embeddings"):
        Camelyon16Adapter(root, embeddings_root=tmp_path / excluded)


def test_embedding_template_rejects_absolute_and_parent_traversal(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    parent = Camelyon16Adapter(root, directory_template="../outside/{magnification}")
    with pytest.raises(ValueError, match="traversal"):
        parent.embedding_location("normal_001", 2.5)

    absolute = Camelyon16Adapter(root, directory_template="/tmp/{magnification}")
    with pytest.raises(ValueError, match="relative"):
        absolute.embedding_location("normal_001", 2.5)


def test_pean_dataset_root_is_rejected_without_discovery(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PEAN"):
        Camelyon16Adapter(tmp_path / "PEAN")
