from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from rl4xs_wsi.data.base import DatasetAdapter, EmbeddingLocation, SlideRecord
from rl4xs_wsi.data.embeddings.hdf5 import AtlasPatchHDF5Reader
from rl4xs_wsi.data.validation import DatasetValidator, ValidationOptions


class MiniAdapter(DatasetAdapter):
    def __init__(self, root: Path, slide_ids: tuple[str, ...]) -> None:
        self._root = root
        self._records = {
            slide_id: SlideRecord(
                slide_id=slide_id,
                slide_path=root / f"{slide_id}.tif",
                label=index % 2,
                label_name="normal" if index % 2 == 0 else "tumor",
                official_cohort="development",
            )
            for index, slide_id in enumerate(slide_ids)
        }

    @property
    def name(self) -> str:
        return "mini"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def magnifications(self) -> tuple[float, ...]:
        return (2.5,)

    def iter_slides(self) -> Iterator[SlideRecord]:
        return iter(self._records.values())

    def get_slide(self, slide_id: str) -> SlideRecord:
        return self._records[slide_id]

    def embedding_location(self, slide_id: str, magnification: float) -> EmbeddingLocation:
        return EmbeddingLocation(
            slide_id=slide_id,
            magnification=magnification,
            path=self._root / f"{slide_id}.h5",
            encoder="test",
            feature_key="features/test",
        )


class CountingReader(AtlasPatchHDF5Reader):
    def __init__(self) -> None:
        self.scans = 0

    def iter_feature_chunks(self, location: EmbeddingLocation, chunk_rows: int):
        self.scans += 1
        yield from super().iter_feature_chunks(location, chunk_rows)


def _write_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=np.asarray([[0, 0, 0, 0, 0]], dtype=np.int32))
        handle.create_dataset("passports", data=np.asarray([b"p"], dtype="S1"))
        handle.create_dataset("features/test", data=np.ones((1, 3), dtype=np.float32))
        handle.attrs.update(
            patch_size=256,
            patch_size_level0=4096,
            level0_width=8192,
            level0_height=8192,
            level0_magnification=40.0,
            target_magnification=2.5,
            overlap=0,
        )


def test_validation_checkpoints_resumes_and_invalidates_changed_file(tmp_path: Path) -> None:
    adapter = MiniAdapter(tmp_path, ("normal_001",))
    _write_h5(tmp_path / "normal_001.h5")
    report_path = tmp_path / "validation.json"
    reader = CountingReader()
    validator = DatasetValidator(
        adapter,
        reader=reader,
        options=ValidationOptions(
            check_wsi_existence=False,
            check_wsi_readability=False,
            feature_chunk_rows=1,
        ),
    )

    first = validator.validate(report_path=report_path)
    assert first.complete
    assert reader.scans == 1
    assert report_path.is_file()

    second = validator.validate(report_path=report_path, resume=True)
    assert second.complete
    assert reader.scans == 1

    path = tmp_path / "normal_001.h5"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    validator.validate(report_path=report_path, resume=True)
    assert reader.scans == 2


def test_selected_development_ids_do_not_touch_other_embeddings_or_wsis(tmp_path: Path) -> None:
    adapter = MiniAdapter(tmp_path, ("normal_001", "test_001"))
    _write_h5(tmp_path / "normal_001.h5")
    # test_001 deliberately has neither WSI nor HDF5.
    report = DatasetValidator(
        adapter,
        options=ValidationOptions(
            check_wsi_existence=False,
            check_wsi_readability=False,
            check_feature_finiteness=False,
        ),
    ).validate(slide_ids=("normal_001",))

    assert report.complete
    assert set(report.per_file) == {"normal_001@wsi", "normal_001@2.5x"}
    assert not any(issue.slide_id == "test_001" for issue in report.issues)
