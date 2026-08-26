from __future__ import annotations

import numpy as np
import pytest

from rl4xs_wsi.errors import PreprocessingError
from rl4xs_wsi.preprocessing.geometry import GeometryBuilder
from rl4xs_wsi.preprocessing.mapping import map_embeddings_to_grid


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
