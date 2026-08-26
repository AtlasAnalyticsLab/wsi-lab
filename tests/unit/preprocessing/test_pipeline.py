from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from wsi_lab.constants import CELL_ID_SLIDE_BLOCK_SIZE
from wsi_lab.data.base import DatasetAdapter, EmbeddingLocation, SlideRecord
from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
from wsi_lab.data.splits import DatasetSplit, SplitAssignment
from wsi_lab.errors import PreprocessingError
from wsi_lab.preprocessing.artifacts import ArtifactStore
from wsi_lab.preprocessing.pipeline import (
    PreprocessingOptions,
    Preprocessor,
    ProcessedEpisodeSource,
)
from wsi_lab.wsi.reader import PillowSlideReader


class MiniAdapter(DatasetAdapter):
    def __init__(self, root: Path, slide_ids: tuple[str, ...]) -> None:
        self._root = root
        self._records: dict[str, SlideRecord] = {}
        for slide_id in slide_ids:
            is_normal = slide_id.startswith("normal_")
            self._records[slide_id] = SlideRecord(
                slide_id=slide_id,
                slide_path=root / f"{slide_id}.png",
                label=0 if is_normal else 1,
                label_name="normal" if is_normal else "tumor",
                official_cohort=("test" if slide_id.startswith("test_") else "development"),
                tissue_mask_path=root / f"{slide_id}_tissue.png",
            )

    @property
    def name(self) -> str:
        return "mini"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def magnifications(self) -> tuple[float, ...]:
        return (2.5, 5.0, 10.0, 20.0)

    def iter_slides(self) -> Iterator[SlideRecord]:
        return iter(self._records.values())

    def get_slide(self, slide_id: str) -> SlideRecord:
        return self._records[slide_id]

    def embedding_location(self, slide_id: str, magnification: float) -> EmbeddingLocation:
        return EmbeddingLocation(
            slide_id=slide_id,
            magnification=magnification,
            path=self._root / f"{slide_id}_{magnification:g}.h5",
            encoder="uni_v2",
            feature_key="features/uni_v2",
        )


def _write_embedding(path: Path, slide_id: str, magnification: float) -> None:
    footprint = {2.5: 4096, 5.0: 2048, 10.0: 1024, 20.0: 512}[magnification]
    if magnification == 20:
        coordinates = np.asarray([[512, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int32)
        features = np.asarray([[20, 0], [20, 1]], dtype=np.float32)
    else:
        coordinates = np.asarray([[0, 0, 0, 0, 0]], dtype=np.int32)
        features = np.asarray([[magnification, 0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=coordinates)
        handle.create_dataset("passports", data=np.asarray([b"p"] * len(coordinates), dtype="S1"))
        handle.create_dataset("features/uni_v2", data=features)
        handle.attrs.update(
            slide_id=slide_id,
            patch_size=256,
            patch_size_level0=footprint,
            level0_width=900,
            level0_height=400,
            level0_magnification=40.0,
            target_magnification=magnification,
            overlap=0,
            mpp=0.25,
        )


def _fixture(tmp_path: Path) -> tuple[MiniAdapter, HDF5EmbeddingProvider]:
    slide_ids = ("normal_001", "tumor_001")
    adapter = MiniAdapter(tmp_path, slide_ids)
    for index, slide_id in enumerate(slide_ids):
        Image.new("RGB", (900, 400), (80 + index * 30, 20, 30)).save(tmp_path / f"{slide_id}.png")
        tissue = np.zeros((40, 90), dtype=np.uint8)
        tissue[:, :45] = 255
        Image.fromarray(tissue).save(tmp_path / f"{slide_id}_tissue.png")
        for magnification in adapter.magnifications:
            _write_embedding(
                adapter.embedding_location(slide_id, magnification).path,
                slide_id,
                magnification,
            )
    return adapter, HDF5EmbeddingProvider(adapter)


def test_preprocessor_persists_global_offsets_sparse_rows_and_mpp_override(tmp_path: Path) -> None:
    adapter, provider = _fixture(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    processor = Preprocessor(
        adapter,
        provider,
        PillowSlideReader(),
        store,
        options=PreprocessingOptions(
            overview_size=32,
            tissue_source="existing",
            tissue_cache_root=tmp_path / "tissue-cache",
        ),
    )

    results = processor.preprocess()

    assert len(results) == 2
    assert results[0].cell_id_offset == 0
    assert results[1].cell_id_offset == CELL_ID_SLIDE_BLOCK_SIZE
    first_ids = set(store.load_geometry("normal_001").level(20).cells["cell_id"].tolist())
    second_ids = set(store.load_geometry("tumor_001").level(20).cells["cell_id"].tolist())
    assert first_ids.isdisjoint(second_ids)
    assert (tmp_path / "tissue-cache" / "mpp" / "normal_001.csv").is_file()
    assert processor.prepare_tissue_mask("normal_001").source == "existing"
    assert processor.preprocess_slide("normal_001").reused
    summary = results[0].mapping_summaries[20.0]
    assert summary.tissue_overlap_fraction is not None
    assert 0.0 <= summary.tissue_overlap_fraction <= 1.0


def test_development_preprocessing_does_not_open_test_embeddings_and_test_extension_is_explicit(
    tmp_path: Path,
) -> None:
    adapter = MiniAdapter(tmp_path, ("normal_001", "test_001"))
    Image.new("RGB", (900, 400), "white").save(tmp_path / "normal_001.png")
    Image.fromarray(np.zeros((40, 90), dtype=np.uint8)).save(tmp_path / "normal_001_tissue.png")
    for magnification in adapter.magnifications:
        _write_embedding(
            adapter.embedding_location("normal_001", magnification).path,
            "normal_001",
            magnification,
        )
    processor = Preprocessor(
        adapter,
        HDF5EmbeddingProvider(adapter),
        PillowSlideReader(),
        ArtifactStore(tmp_path / "artifacts"),
        options=PreprocessingOptions(
            overview_size=32,
            tissue_source="existing",
            tissue_cache_root=tmp_path / "tissue-cache",
        ),
    )

    mask = processor.prepare_tissue_mask("normal_001")
    development = processor.preprocess(("normal_001",))

    assert mask.source == "existing"
    assert len(development) == 1
    assert tuple(processor.artifacts.load_offsets()) == ("normal_001",)
    assert not adapter.embedding_location("test_001", 2.5).path.exists()

    Image.new("RGB", (900, 400), "white").save(tmp_path / "test_001.png")
    Image.fromarray(np.zeros((40, 90), dtype=np.uint8)).save(tmp_path / "test_001_tissue.png")
    for magnification in adapter.magnifications:
        _write_embedding(
            adapter.embedding_location("test_001", magnification).path,
            "test_001",
            magnification,
        )
    official_test = processor.preprocess(("test_001",), merge_index=True)

    offsets = processor.artifacts.load_offsets()
    assert len(official_test) == 1
    assert tuple(offsets) == ("normal_001", "test_001")
    assert offsets["test_001"].offset == CELL_ID_SLIDE_BLOCK_SIZE


def test_sparse_offset_allocation_skips_interleaved_unrequested_test_cohort(
    tmp_path: Path,
) -> None:
    adapter = MiniAdapter(tmp_path, ("normal_001", "test_001", "tumor_001"))
    for slide_id in ("normal_001", "tumor_001"):
        Image.new("RGB", (900, 400), "white").save(tmp_path / f"{slide_id}.png")
        Image.fromarray(np.zeros((40, 90), dtype=np.uint8)).save(
            tmp_path / f"{slide_id}_tissue.png"
        )
        for magnification in adapter.magnifications:
            _write_embedding(
                adapter.embedding_location(slide_id, magnification).path,
                slide_id,
                magnification,
            )
    processor = Preprocessor(
        adapter,
        HDF5EmbeddingProvider(adapter),
        PillowSlideReader(),
        ArtifactStore(tmp_path / "artifacts"),
        options=PreprocessingOptions(
            overview_size=32,
            tissue_source="existing",
            tissue_cache_root=tmp_path / "tissue-cache",
        ),
    )

    results = processor.preprocess(("normal_001", "tumor_001"))

    assert tuple(result.slide_id for result in results) == ("normal_001", "tumor_001")
    offsets = processor.artifacts.load_offsets()
    assert tuple(offsets) == ("normal_001", "tumor_001")
    assert offsets["normal_001"].offset == 0
    assert offsets["tumor_001"].offset == 2 * CELL_ID_SLIDE_BLOCK_SIZE
    assert not adapter.embedding_location("test_001", 2.5).path.exists()


def test_cell_count_must_fit_fixed_slide_block(tmp_path: Path) -> None:
    adapter = MiniAdapter(tmp_path, ("normal_001",))

    class OversizedProvider:
        def metadata(self, slide_id: str, magnification: float):
            del slide_id, magnification
            return type(
                "Metadata",
                (),
                {"level0_width": 600_000_000, "level0_height": 600_000_000},
            )()

    processor = Preprocessor(
        adapter,
        OversizedProvider(),  # type: ignore[arg-type]
        PillowSlideReader(),
        ArtifactStore(tmp_path / "artifacts"),
    )

    with pytest.raises(PreprocessingError, match="cell-ID block capacity"):
        processor.preprocess(("normal_001",))


def test_processed_episode_source_preserves_letterbox_and_hdf5_source_row(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    adapter, provider = _fixture(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    processor = Preprocessor(
        adapter,
        provider,
        PillowSlideReader(),
        store,
        options=PreprocessingOptions(
            overview_size=32,
            tissue_source="existing",
            tissue_cache_root=tmp_path / "tissue-cache",
        ),
    )
    processor.preprocess()
    split = DatasetSplit(
        dataset="mini",
        seed=1,
        validation_fraction=0.5,
        strategy="slide",
        assignments=(
            SplitAssignment("normal_001", 0, "train"),
            SplitAssignment("tumor_001", 1, "val"),
        ),
    )
    source = ProcessedEpisodeSource(adapter, provider, PillowSlideReader(), store, split)

    loaded = source.load_slide("normal_001")
    assert loaded.overview_transform.content_width == 32
    assert loaded.overview_transform.content_height == 14
    assert loaded.overview_transform.content_y0 == 9
    spatial = np.zeros((32, 32), dtype=np.float32)
    x0, y0, x1, y1 = loaded.overview_transform.raster_bounds(
        loaded.cells[loaded.start_cell_id].bounds
    )
    spatial[y0:y1, x0:x1] = 1
    assert not spatial[: loaded.overview_transform.content_y0].any()

    grid = store.load_grid("normal_001", 20)
    left = loaded.cells[int(grid["cell_id"][0])]
    assert left.embedding_index == 1
    assert source.read_embedding("normal_001", left).tolist() == [20.0, 1.0]
    right = loaded.cells[int(grid["cell_id"][1])]
    viewport = source.read_viewport("normal_001", right, 32)
    assert viewport.shape == (32, 32, 3)
    assert np.all(viewport[:, -1] >= 250)  # nominal 512px cell overhang is padded white
