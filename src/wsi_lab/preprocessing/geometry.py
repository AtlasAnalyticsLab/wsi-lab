"""Embedding-independent six-level navigation geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from wsi_lab.constants import LEVEL0_FOOTPRINTS, NAVIGATION_MAGNIFICATIONS
from wsi_lab.errors import PreprocessingError

GRID_DTYPE = np.dtype(
    [
        ("cell_id", "<i8"),
        ("slide_index", "<i4"),
        ("magnification", "<f4"),
        ("row", "<i4"),
        ("column", "<i4"),
        ("level0_x0", "<i8"),
        ("level0_y0", "<i8"),
        # Nominal bounds preserve exact quadtree footprints and can exceed a
        # slide edge. Clipped bounds describe the actual valid WSI rectangle.
        ("level0_x1", "<i8"),
        ("level0_y1", "<i8"),
        ("clipped_x1", "<i8"),
        ("clipped_y1", "<i8"),
        ("parent_id", "<i8"),
        ("child_tl_id", "<i8"),
        ("child_tr_id", "<i8"),
        ("child_bl_id", "<i8"),
        ("child_br_id", "<i8"),
        ("neighbor_up", "<i8"),
        ("neighbor_down", "<i8"),
        ("neighbor_left", "<i8"),
        ("neighbor_right", "<i8"),
        ("tissue_fraction", "<f4"),
        ("tissue_status", "u1"),
        ("embedding_index", "<i8"),
    ]
)


@dataclass(frozen=True, slots=True)
class GridLevel:
    magnification: float
    footprint_level0: int
    rows: int
    columns: int
    offset: int
    cells: NDArray[np.void]

    def local_index(self, row: int, column: int) -> int:
        if not 0 <= row < self.rows or not 0 <= column < self.columns:
            raise IndexError((row, column))
        return row * self.columns + column

    def cell_id(self, row: int, column: int) -> int:
        return self.offset + self.local_index(row, column)

    def cell(self, row: int, column: int) -> np.void:
        return self.cells[self.local_index(row, column)]


@dataclass(frozen=True, slots=True)
class NavigationGeometry:
    slide_id: str
    slide_index: int
    level0_width: int
    level0_height: int
    level0_magnification: float
    cell_id_offset: int
    levels: Mapping[float, GridLevel]

    def level(self, magnification: float) -> GridLevel:
        try:
            return self.levels[float(magnification)]
        except KeyError as exc:
            raise KeyError(f"unknown navigation magnification {magnification:g}") from exc

    @property
    def magnifications(self) -> tuple[float, ...]:
        return tuple(self.levels)

    @property
    def num_cells(self) -> int:
        return sum(len(level.cells) for level in self.levels.values())


class GeometryBuilder:
    """Build a clipped-edge quadtree with row-major stable global cell IDs."""

    def __init__(
        self,
        *,
        magnifications: tuple[float, ...] = NAVIGATION_MAGNIFICATIONS,
        footprints: Mapping[float, int] = LEVEL0_FOOTPRINTS,
    ) -> None:
        self.magnifications = tuple(float(value) for value in magnifications)
        self.footprints = {float(key): int(value) for key, value in footprints.items()}
        self._validate_protocol()

    def _validate_protocol(self) -> None:
        if not self.magnifications:
            raise ValueError("navigation magnifications cannot be empty")
        if len(set(self.magnifications)) != len(self.magnifications):
            raise ValueError("navigation magnifications must be unique")
        if tuple(sorted(self.magnifications)) != self.magnifications:
            raise ValueError("navigation magnifications must be increasing")
        missing = set(self.magnifications).difference(self.footprints)
        if missing:
            raise ValueError(f"missing level-0 footprints for magnifications {sorted(missing)}")
        for lower, higher in pairwise(self.magnifications):
            if not np.isclose(higher / lower, 2.0):
                raise ValueError("adjacent navigation magnifications must differ by exactly 2x")
            if self.footprints[lower] != 2 * self.footprints[higher]:
                raise ValueError("adjacent navigation footprints must halve exactly")

    def build(
        self,
        slide_id: str,
        *,
        level0_width: int,
        level0_height: int,
        level0_magnification: float,
        cell_id_offset: int = 0,
        slide_index: int = 0,
    ) -> NavigationGeometry:
        if level0_width <= 0 or level0_height <= 0:
            raise ValueError("level-0 slide dimensions must be positive")
        if cell_id_offset < 0:
            raise ValueError("cell_id_offset cannot be negative")
        if slide_index < 0:
            raise ValueError("slide_index cannot be negative")
        shapes: dict[float, tuple[int, int, int]] = {}
        offset = int(cell_id_offset)
        for magnification in self.magnifications:
            footprint = self.footprints[magnification]
            rows = math.ceil(level0_height / footprint)
            columns = math.ceil(level0_width / footprint)
            shapes[magnification] = (rows, columns, offset)
            offset += rows * columns
        levels: dict[float, GridLevel] = {}
        for level_index, magnification in enumerate(self.magnifications):
            footprint = self.footprints[magnification]
            rows, columns, offset = shapes[magnification]
            cells = np.empty(rows * columns, dtype=GRID_DTYPE)
            cells["parent_id"] = -1
            for field in ("child_tl_id", "child_tr_id", "child_bl_id", "child_br_id"):
                cells[field] = -1
            for field in ("neighbor_up", "neighbor_down", "neighbor_left", "neighbor_right"):
                cells[field] = -1
            cells["embedding_index"] = -1
            cells["tissue_fraction"] = np.nan
            cells["tissue_status"] = 0
            local = np.arange(rows * columns, dtype=np.int64)
            row = local // columns
            column = local % columns
            x0 = column * footprint
            y0 = row * footprint
            cells["cell_id"] = offset + local
            cells["slide_index"] = slide_index
            cells["magnification"] = magnification
            cells["row"] = row
            cells["column"] = column
            cells["level0_x0"] = x0
            cells["level0_y0"] = y0
            cells["level0_x1"] = x0 + footprint
            cells["level0_y1"] = y0 + footprint
            cells["clipped_x1"] = np.minimum(x0 + footprint, level0_width)
            cells["clipped_y1"] = np.minimum(y0 + footprint, level0_height)
            has_up = row > 0
            has_down = row + 1 < rows
            has_left = column > 0
            has_right = column + 1 < columns
            cells["neighbor_up"][has_up] = offset + local[has_up] - columns
            cells["neighbor_down"][has_down] = offset + local[has_down] + columns
            cells["neighbor_left"][has_left] = offset + local[has_left] - 1
            cells["neighbor_right"][has_right] = offset + local[has_right] + 1
            if level_index > 0:
                previous = self.magnifications[level_index - 1]
                parent_rows, parent_columns, parent_offset = shapes[previous]
                parent_row = row // 2
                parent_column = column // 2
                if np.any(parent_row >= parent_rows) or np.any(parent_column >= parent_columns):
                    raise PreprocessingError("edge geometry produced an invalid parent")
                cells["parent_id"] = parent_offset + parent_row * parent_columns + parent_column
            if level_index + 1 < len(self.magnifications):
                following = self.magnifications[level_index + 1]
                child_rows, child_columns, child_offset = shapes[following]
                for field, dy, dx in (
                    ("child_tl_id", 0, 0),
                    ("child_tr_id", 0, 1),
                    ("child_bl_id", 1, 0),
                    ("child_br_id", 1, 1),
                ):
                    child_row = row * 2 + dy
                    child_column = column * 2 + dx
                    valid = (child_row < child_rows) & (child_column < child_columns)
                    cells[field][valid] = (
                        child_offset + child_row[valid] * child_columns + child_column[valid]
                    )
            levels[magnification] = GridLevel(
                magnification=magnification,
                footprint_level0=footprint,
                rows=rows,
                columns=columns,
                offset=offset,
                cells=cells,
            )
        result = NavigationGeometry(
            slide_id=slide_id,
            slide_index=int(slide_index),
            level0_width=int(level0_width),
            level0_height=int(level0_height),
            level0_magnification=float(level0_magnification),
            cell_id_offset=int(cell_id_offset),
            levels=levels,
        )
        validate_geometry(result)
        return result


def validate_geometry(geometry: NavigationGeometry) -> None:
    expected_id = geometry.cell_id_offset
    magnifications = geometry.magnifications
    for index, magnification in enumerate(magnifications):
        level = geometry.level(magnification)
        if level.offset != expected_id or len(level.cells) != level.rows * level.columns:
            raise PreprocessingError(f"invalid grid shape or offset at {magnification:g}x")
        ids = level.cells["cell_id"]
        if not np.array_equal(ids, np.arange(expected_id, expected_id + len(level.cells))):
            raise PreprocessingError(f"non-contiguous cell IDs at {magnification:g}x")
        if np.any(level.cells["slide_index"] != geometry.slide_index):
            raise PreprocessingError(f"grid slide identifier mismatch at {magnification:g}x")
        if np.any(level.cells["level0_x0"] >= geometry.level0_width) or np.any(
            level.cells["level0_y0"] >= geometry.level0_height
        ):
            raise PreprocessingError(
                f"grid contains an origin outside the WSI at {magnification:g}x"
            )
        if index > 0:
            parent = geometry.level(magnifications[index - 1])
            parent_ids = level.cells["parent_id"]
            if np.any(
                (parent_ids < parent.offset) | (parent_ids >= parent.offset + len(parent.cells))
            ):
                raise PreprocessingError(f"invalid parent IDs at {magnification:g}x")
        expected_id += len(level.cells)
    # Verify every stored child has a matching round-trip parent.
    for magnification in magnifications[:-1]:
        level = geometry.level(magnification)
        following = geometry.level(magnifications[magnifications.index(magnification) + 1])
        for field in ("child_tl_id", "child_tr_id", "child_bl_id", "child_br_id"):
            child_ids = level.cells[field]
            valid = child_ids >= 0
            local = child_ids[valid] - following.offset
            if np.any(following.cells["parent_id"][local] != level.cells["cell_id"][valid]):
                raise PreprocessingError(
                    f"parent/child round-trip failed for {field} at {magnification:g}x"
                )


__all__ = [
    "GRID_DTYPE",
    "GeometryBuilder",
    "GridLevel",
    "NavigationGeometry",
    "validate_geometry",
]
