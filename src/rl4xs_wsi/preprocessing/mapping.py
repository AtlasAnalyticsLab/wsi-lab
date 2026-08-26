"""Attach sparse embedding rows to independently generated grid cells."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl4xs_wsi.errors import PreprocessingError
from rl4xs_wsi.preprocessing.geometry import GridLevel


@dataclass(frozen=True, slots=True)
class MappingSummary:
    input_rows: int
    mapped_rows: int
    mapped_cells: int
    empty_cells: int
    exact_duplicate_groups: int
    duplicate_extra_rows: int
    conflicting_cell_collisions: int
    off_grid_rows: int
    out_of_bounds_rows: int
    boundary_overhang_rows: int
    tissue_overlapping_cells: int | None = None
    tissue_overlap_fraction: float | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingMapping:
    magnification: float
    source_by_local_cell: NDArray[np.int64]
    duplicate_source_groups: tuple[tuple[int, ...], ...]
    summary: MappingSummary

    def attach(self, level: GridLevel, *, copy: bool = True) -> NDArray[np.void]:
        if len(level.cells) != len(self.source_by_local_cell):
            raise ValueError("mapping length does not match grid level")
        target = level.cells.copy() if copy else level.cells
        target["embedding_index"] = self.source_by_local_cell
        return target


def map_embeddings_to_grid(
    level: GridLevel,
    coordinates: NDArray[np.integer],
    *,
    source_indices: Sequence[int] | None = None,
    tolerance: int = 0,
    level0_width: int | None = None,
    level0_height: int | None = None,
    strict: bool = True,
) -> EmbeddingMapping:
    """Map coordinate origins, retaining the lowest row for exact duplicates.

    Feature/passport equivalence for exact duplicate coordinates belongs to the
    validation stage. A collision from *different* raw coordinates is always
    treated as ambiguous and is fatal in strict mode.
    """

    array = np.asarray(coordinates)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"coordinates must have shape [N,C>=2], got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("coordinates must use an integer dtype")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    count = len(array)
    source = (
        np.arange(count, dtype=np.int64)
        if source_indices is None
        else np.asarray(source_indices, dtype=np.int64)
    )
    if source.ndim != 1 or len(source) != count:
        raise ValueError("source_indices must have shape [N]")
    if len(np.unique(source)) != len(source):
        raise ValueError("source_indices must be unique")
    xy = array[:, :2].astype(np.int64, copy=False)
    footprint = int(level.footprint_level0)
    columns = np.rint(xy[:, 0] / footprint).astype(np.int64)
    rows = np.rint(xy[:, 1] / footprint).astype(np.int64)
    expected_x = columns * footprint
    expected_y = rows * footprint
    off_grid = (np.abs(xy[:, 0] - expected_x) > tolerance) | (
        np.abs(xy[:, 1] - expected_y) > tolerance
    )
    out_of_bounds = (rows < 0) | (columns < 0) | (rows >= level.rows) | (columns >= level.columns)
    if level0_width is not None:
        out_of_bounds |= xy[:, 0] >= level0_width
    if level0_height is not None:
        out_of_bounds |= xy[:, 1] >= level0_height
    valid = ~off_grid & ~out_of_bounds
    boundary = np.zeros(count, dtype=np.bool_)
    if level0_width is not None:
        boundary |= xy[:, 0] + footprint > level0_width
    if level0_height is not None:
        boundary |= xy[:, 1] + footprint > level0_height
    boundary &= valid
    local = rows * level.columns + columns
    selected = np.full(len(level.cells), -1, dtype=np.int64)
    coordinate_by_cell: dict[int, tuple[int, int]] = {}
    sources_by_cell: dict[int, list[int]] = {}
    conflicts = 0
    mapped_rows = 0
    for input_index in np.flatnonzero(valid):
        local_index = int(local[input_index])
        raw_coordinate = (int(xy[input_index, 0]), int(xy[input_index, 1]))
        source_index = int(source[input_index])
        previous_coordinate = coordinate_by_cell.get(local_index)
        if previous_coordinate is None:
            coordinate_by_cell[local_index] = raw_coordinate
            sources_by_cell[local_index] = [source_index]
            selected[local_index] = source_index
            mapped_rows += 1
        elif previous_coordinate == raw_coordinate:
            sources_by_cell[local_index].append(source_index)
            selected[local_index] = min(selected[local_index], source_index)
            mapped_rows += 1
        else:
            conflicts += 1
    duplicate_groups = tuple(
        tuple(sorted(values)) for _, values in sorted(sources_by_cell.items()) if len(values) > 1
    )
    off_grid_count = int(np.count_nonzero(off_grid))
    out_of_bounds_count = int(np.count_nonzero(out_of_bounds & ~off_grid))
    if strict and (off_grid_count or out_of_bounds_count or conflicts):
        raise PreprocessingError(
            f"embedding mapping failed at {level.magnification:g}x: off_grid={off_grid_count}, "
            f"out_of_bounds={out_of_bounds_count}, conflicting_collisions={conflicts}"
        )
    mapped_cells = int(np.count_nonzero(selected >= 0))
    summary = MappingSummary(
        input_rows=count,
        mapped_rows=mapped_rows,
        mapped_cells=mapped_cells,
        empty_cells=len(selected) - mapped_cells,
        exact_duplicate_groups=len(duplicate_groups),
        duplicate_extra_rows=sum(len(group) - 1 for group in duplicate_groups),
        conflicting_cell_collisions=conflicts,
        off_grid_rows=off_grid_count,
        out_of_bounds_rows=out_of_bounds_count,
        boundary_overhang_rows=int(np.count_nonzero(boundary)),
    )
    return EmbeddingMapping(
        magnification=level.magnification,
        source_by_local_cell=selected,
        duplicate_source_groups=duplicate_groups,
        summary=summary,
    )


__all__ = ["EmbeddingMapping", "MappingSummary", "map_embeddings_to_grid"]
