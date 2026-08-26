from __future__ import annotations

import numpy as np
import pytest

from wsi_lab.constants import CELL_ID_SLIDE_BLOCK_SIZE
from wsi_lab.data.camelyon16 import Camelyon16Adapter
from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
from wsi_lab.preprocessing.artifacts import ArtifactStore
from wsi_lab.preprocessing.geometry import GeometryBuilder
from wsi_lab.preprocessing.mapping import map_embeddings_to_grid
from wsi_lab.preprocessing.pipeline import PreprocessingOptions, Preprocessor
from wsi_lab.preprocessing.tissue import load_or_create_tissue_mask
from wsi_lab.wsi.reader import OpenSlideReader, OverviewTransform

pytestmark = pytest.mark.real


@pytest.mark.parametrize(
    ("slide_id", "magnification", "coordinate"),
    [
        ("normal_031", 10.0, (40960, 100352)),
        ("tumor_017", 2.5, (57344, 69632)),
        ("tumor_075", 2.5, (16384, 73728)),
    ],
)
def test_canonical_uni_duplicate_coordinates_collapse_to_lowest_source_row(
    camelyon16_root, slide_id: str, magnification: float, coordinate: tuple[int, int]
) -> None:
    adapter = Camelyon16Adapter(camelyon16_root)
    provider = HDF5EmbeddingProvider(adapter)
    metadata = provider.metadata(slide_id, magnification)
    coordinates = provider.coordinates(slide_id, magnification)
    geometry = GeometryBuilder().build(
        slide_id,
        level0_width=metadata.level0_width,
        level0_height=metadata.level0_height,
        level0_magnification=metadata.level0_magnification,
    )

    rows = np.flatnonzero(np.all(coordinates[:, :2] == coordinate, axis=1))
    mapping = map_embeddings_to_grid(
        geometry.level(magnification),
        coordinates,
        level0_width=metadata.level0_width,
        level0_height=metadata.level0_height,
    )
    local = coordinate[1] // metadata.patch_size_level0 * geometry.level(magnification).columns
    local += coordinate[0] // metadata.patch_size_level0

    assert len(rows) == 2
    assert mapping.source_by_local_cell[local] == int(rows.min())
    assert mapping.summary.exact_duplicate_groups >= 1


def test_real_sparse_child_and_boundary_overhang_are_accepted(camelyon16_root) -> None:
    adapter = Camelyon16Adapter(camelyon16_root)
    provider = HDF5EmbeddingProvider(adapter)
    slide_id = "tumor_017"
    low = provider.metadata(slide_id, 5)
    geometry = GeometryBuilder().build(
        slide_id,
        level0_width=low.level0_width,
        level0_height=low.level0_height,
        level0_magnification=low.level0_magnification,
    )
    child_coordinates = provider.coordinates(slide_id, 10)
    mapping = map_embeddings_to_grid(
        geometry.level(10),
        child_coordinates,
        level0_width=low.level0_width,
        level0_height=low.level0_height,
    )
    attached = mapping.attach(geometry.level(10))
    parent = geometry.level(5).cell(126976 // 2048, 4096 // 2048)
    child_ids = [
        int(parent[field]) for field in ("child_tl_id", "child_tr_id", "child_bl_id", "child_br_id")
    ]
    source_rows = [
        int(attached["embedding_index"][cell_id - geometry.level(10).offset])
        for cell_id in child_ids
    ]
    assert all(cell_id >= 0 for cell_id in child_ids)
    assert [value >= 0 for value in source_rows] == [False, True, True, True]

    boundary_metadata = provider.metadata("normal_031", 2.5)
    boundary_geometry = GeometryBuilder().build(
        "normal_031",
        level0_width=boundary_metadata.level0_width,
        level0_height=boundary_metadata.level0_height,
        level0_magnification=boundary_metadata.level0_magnification,
    )
    boundary_mapping = map_embeddings_to_grid(
        boundary_geometry.level(2.5),
        provider.coordinates("normal_031", 2.5),
        level0_width=boundary_metadata.level0_width,
        level0_height=boundary_metadata.level0_height,
    )
    assert boundary_mapping.summary.boundary_overhang_rows > 0


def test_real_existing_tissue_mask_is_binary_and_transform_compatible(camelyon16_root) -> None:
    adapter = Camelyon16Adapter(camelyon16_root)
    provider = HDF5EmbeddingProvider(adapter)
    record = adapter.get_slide("normal_031")
    metadata = provider.metadata("normal_031", 2.5)
    transform = OverviewTransform.fit((metadata.level0_width, metadata.level0_height), (224, 224))
    overview = np.full((224, 224, 3), 255, dtype=np.uint8)

    mask = load_or_create_tissue_mask(
        record.slide_path,
        overview,
        transform,
        existing_path=record.tissue_mask_path,
        source="existing",
    )

    assert mask.values.dtype == np.bool_
    assert mask.values.shape == (224, 224)
    assert mask.source == "existing"


@pytest.mark.slow
def test_preprocess_exactly_two_real_slides_into_temporary_workspace(
    tmp_path, camelyon16_root
) -> None:
    adapter = Camelyon16Adapter(camelyon16_root)
    provider = HDF5EmbeddingProvider(adapter)
    store = ArtifactStore(tmp_path / "processed")
    reader = OpenSlideReader(handle_cache_size=1, viewport_cache_size=0)
    processor = Preprocessor(
        adapter,
        provider,
        reader,
        store,
        options=PreprocessingOptions(
            overview_size=64,
            tissue_source="auto",
            tissue_cache_root=tmp_path / "tissue",
        ),
    )
    try:
        results = processor.preprocess(("normal_031", "tumor_017"))
    finally:
        reader.close()

    assert tuple(result.slide_id for result in results) == ("normal_031", "tumor_017")
    assert all(result.num_cells > 0 for result in results)
    assert all(store.is_slide_complete(result.slide_id) for result in results)
    positions = {record.slide_id: index for index, record in enumerate(adapter.iter_slides())}
    for result in results:
        assert result.cell_id_offset == positions[result.slide_id] * CELL_ID_SLIDE_BLOCK_SIZE
        for summary in result.mapping_summaries.values():
            assert summary.tissue_overlap_fraction is not None
            assert 0.0 <= summary.tissue_overlap_fraction <= 1.0
