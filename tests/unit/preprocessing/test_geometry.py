from __future__ import annotations

import numpy as np
import pytest

from wsi_lab.errors import PreprocessingError
from wsi_lab.preprocessing.geometry import GeometryBuilder
from wsi_lab.preprocessing.mapping import map_embeddings_to_grid
from wsi_lab.preprocessing.tissue import TissueMask, apply_tissue_to_geometry
from wsi_lab.wsi.reader import OverviewTransform


def test_dataset_global_offsets_are_deterministic_and_noncolliding() -> None:
    builder = GeometryBuilder()
    first = builder.build(
        "slide_a", level0_width=20_000, level0_height=17_000, level0_magnification=40
    )
    second = builder.build(
        "slide_b",
        level0_width=12_000,
        level0_height=10_000,
        level0_magnification=40,
        cell_id_offset=first.num_cells,
    )
    repeated = builder.build(
        "slide_b",
        level0_width=12_000,
        level0_height=10_000,
        level0_magnification=40,
        cell_id_offset=first.num_cells,
    )

    assert int(first.level(20).cells["cell_id"][-1]) < int(second.level(0.625).cells["cell_id"][0])
    assert second.cell_id_offset == first.num_cells
    for magnification in second.magnifications:
        assert np.array_equal(
            second.level(magnification).cells["cell_id"],
            repeated.level(magnification).cells["cell_id"],
        )


def test_boundary_cells_keep_nominal_bounds_and_clipped_bounds() -> None:
    geometry = GeometryBuilder().build(
        "edge", level0_width=17_000, level0_height=9_000, level0_magnification=40
    )
    edge = geometry.level(0.625).cells[-1]

    assert int(edge["level0_x1"]) > geometry.level0_width
    assert int(edge["level0_y1"]) > geometry.level0_height
    assert int(edge["clipped_x1"]) == geometry.level0_width
    assert int(edge["clipped_y1"]) == geometry.level0_height


def test_sparse_embedding_children_are_valid_geometry() -> None:
    geometry = GeometryBuilder().build(
        "sparse", level0_width=4096, level0_height=4096, level0_magnification=40
    )
    parent = geometry.level(5).cell(0, 0)
    children = tuple(
        int(parent[field]) for field in ("child_tl_id", "child_tr_id", "child_bl_id", "child_br_id")
    )
    mapping = map_embeddings_to_grid(
        geometry.level(10),
        np.asarray([[0, 0], [1024, 0], [0, 1024]], dtype=np.int32),
        level0_width=4096,
        level0_height=4096,
    )
    attached = mapping.attach(geometry.level(10))

    assert all(child >= 0 for child in children)
    local_children = [child - geometry.level(10).offset for child in children]
    assert attached["embedding_index"][local_children].tolist() == [0, 1, 2, -1]
    assert mapping.summary.empty_cells > 0


def test_exact_duplicates_choose_lowest_source_row_and_off_grid_fails() -> None:
    level = (
        GeometryBuilder()
        .build("duplicates", level0_width=4096, level0_height=4096, level0_magnification=40)
        .level(2.5)
    )
    mapping = map_embeddings_to_grid(
        level,
        np.asarray([[0, 0], [0, 0]], dtype=np.int32),
        source_indices=[9, 3],
    )
    assert mapping.source_by_local_cell[0] == 3
    assert mapping.duplicate_source_groups == ((3, 9),)

    with pytest.raises(PreprocessingError, match="off_grid=1"):
        map_embeddings_to_grid(level, np.asarray([[1, 0]], dtype=np.int32), strict=True)


def _disconnected_tissue_geometry():
    builder = GeometryBuilder(
        magnifications=(1.0, 2.0, 4.0),
        footprints={1.0: 4, 2.0: 2, 4.0: 1},
    )
    geometry = builder.build(
        "islands",
        level0_width=12,
        level0_height=8,
        level0_magnification=4,
        cell_id_offset=100,
    )
    values = np.zeros((8, 12), dtype=np.bool_)
    values[2, 1] = True
    values[2, 6] = True
    values[6, 10] = True
    mask = TissueMask(values=values, source="test", threshold=0.75)
    transform = OverviewTransform.fit((12, 8), (12, 8))
    return geometry, apply_tissue_to_geometry(geometry, mask, transform)


def test_disconnected_tissue_bbox_and_directional_neighbors_skip_background() -> None:
    _, geometry = _disconnected_tissue_geometry()
    level = geometry.level(4)
    left = level.cell(2, 1)
    right = level.cell(2, 6)
    background = level.cell(2, 3)

    assert geometry.tissue_bbox_level0 == (1, 2, 11, 7)
    assert geometry.navigation_roi_level0 == (0, 0, 12, 8)
    assert int(left["neighbor_right"]) == int(right["cell_id"])
    assert int(right["neighbor_left"]) == int(left["cell_id"])
    assert int(left["neighbor_left"]) == -1
    assert int(right["neighbor_right"]) == -1
    assert all(
        int(background[field]) == -1
        for field in ("neighbor_up", "neighbor_down", "neighbor_left", "neighbor_right")
    )


def test_tissue_attachment_preserves_global_hierarchy_and_embedding_alignment() -> None:
    original, attached = _disconnected_tissue_geometry()
    hierarchy_fields = (
        "cell_id",
        "row",
        "column",
        "level0_x0",
        "level0_y0",
        "level0_x1",
        "level0_y1",
        "parent_id",
        "child_tl_id",
        "child_tr_id",
        "child_bl_id",
        "child_br_id",
    )
    for magnification in original.magnifications:
        for field in hierarchy_fields:
            assert np.array_equal(
                original.level(magnification).cells[field],
                attached.level(magnification).cells[field],
            )

    finest = attached.level(4)
    mapping = map_embeddings_to_grid(
        finest,
        np.asarray([[1, 2]], dtype=np.int32),
        level0_width=attached.level0_width,
        level0_height=attached.level0_height,
        strict=True,
    )
    cells = mapping.attach(finest)
    embedded_tissue = finest.local_index(2, 1)
    missing_embedding_tissue = finest.local_index(2, 6)
    assert int(cells["embedding_index"][embedded_tissue]) == 0
    assert bool(cells["tissue_valid"][missing_embedding_tissue])
    assert int(cells["embedding_index"][missing_embedding_tissue]) == -1

    # Any overlap is navigable even when the configured legacy threshold is
    # much higher than the cell's tissue fraction.
    coarse = attached.level(1).cell(0, 0)
    assert bool(coarse["tissue_valid"])
    assert not bool(coarse["tissue_status"])

    # The four child IDs remain geometric; their tissue-valid flags provide
    # the topology needed to mask the three empty zoom-in quadrants.
    parent = attached.level(2).cell(1, 0)
    child_ids = [
        int(parent[field]) for field in ("child_tl_id", "child_tr_id", "child_bl_id", "child_br_id")
    ]
    child_local = np.asarray(child_ids) - attached.level(4).offset
    assert attached.level(4).cells["tissue_valid"][child_local].tolist() == [0, 1, 0, 0]


def test_empty_tissue_mask_cannot_build_navigation_topology() -> None:
    builder = GeometryBuilder(
        magnifications=(1.0, 2.0),
        footprints={1.0: 2, 2.0: 1},
    )
    geometry = builder.build(
        "empty",
        level0_width=4,
        level0_height=4,
        level0_magnification=2,
    )
    transform = OverviewTransform.fit((4, 4), (4, 4))
    mask = TissueMask(
        values=np.zeros((4, 4), dtype=np.bool_),
        source="test",
        threshold=0.01,
    )

    with pytest.raises(PreprocessingError, match="no tissue pixels"):
        apply_tissue_to_geometry(geometry, mask, transform)


def test_tissue_mask_shape_must_match_overview_canvas() -> None:
    geometry = GeometryBuilder(
        magnifications=(1.0, 2.0),
        footprints={1.0: 2, 2.0: 1},
    ).build(
        "misaligned",
        level0_width=4,
        level0_height=4,
        level0_magnification=2,
    )
    transform = OverviewTransform.fit((4, 4), (4, 4))
    mask = TissueMask(
        values=np.ones((3, 4), dtype=np.bool_),
        source="test",
        threshold=0.01,
    )

    with pytest.raises(PreprocessingError, match=r"mask shape.*overview canvas"):
        apply_tissue_to_geometry(geometry, mask, transform)
