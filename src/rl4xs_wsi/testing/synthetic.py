"""Deterministic installed fixture used by pytest and the smoke-test command."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl4xs_wsi.constants import LEVEL0_FOOTPRINTS, NAVIGATION_MAGNIFICATIONS
from rl4xs_wsi.envs.wsi_env import LoadedSlide, NavigationCell
from rl4xs_wsi.wsi.reader import OverviewTransform


class DeterministicBelief:
    """Simple evidence-dependent probability for environment contract tests."""

    def __init__(self) -> None:
        self.reset()

    @property
    def probability(self) -> float:
        return self._probability

    def reset(self) -> None:
        self._logit = 0.0
        self._probability = 0.5

    def add(self, feature: np.ndarray, magnification: float) -> float:
        del magnification
        self._logit += float(np.asarray(feature).mean())
        self._probability = float(1.0 / (1.0 + np.exp(-self._logit)))
        return self._probability


@dataclass
class SyntheticEpisodeSource:
    overview_size: int = 224
    level0_size: int = 32768

    def __post_init__(self) -> None:
        self._slides = {
            "synthetic_benign": self._build_slide("synthetic_benign", 0),
            "synthetic_cancer": self._build_slide("synthetic_cancer", 1),
        }

    def slide_ids(self, split: str):
        del split
        return tuple(self._slides)

    def _build_slide(self, slide_id: str, label: int) -> LoadedSlide:
        canvas = self.overview_size
        y, x = np.mgrid[:canvas, :canvas]
        overview = np.stack(
            (
                (x * 255 // max(1, canvas - 1)),
                (y * 255 // max(1, canvas - 1)),
                np.full_like(x, 80 + label * 80),
            ),
            axis=-1,
        ).astype(np.uint8)
        transform = OverviewTransform.fit((self.level0_size, self.level0_size), (canvas, canvas))
        cells: dict[int, NavigationCell] = {}
        ids_by_level: dict[float, np.ndarray] = {}
        next_id = 0
        for magnification in NAVIGATION_MAGNIFICATIONS:
            footprint = LEVEL0_FOOTPRINTS[magnification]
            rows = int(np.ceil(self.level0_size / footprint))
            columns = rows
            identifiers = np.arange(next_id, next_id + rows * columns, dtype=np.int64).reshape(
                rows, columns
            )
            next_id += rows * columns
            ids_by_level[magnification] = identifiers
        embedding_counter = 0
        for level_index, magnification in enumerate(NAVIGATION_MAGNIFICATIONS):
            identifiers = ids_by_level[magnification]
            footprint = LEVEL0_FOOTPRINTS[magnification]
            rows, columns = identifiers.shape
            for row in range(rows):
                for column in range(columns):
                    cell_id = int(identifiers[row, column])
                    parent = -1
                    if level_index:
                        parent = int(
                            ids_by_level[NAVIGATION_MAGNIFICATIONS[level_index - 1]][
                                row // 2, column // 2
                            ]
                        )
                    children = (-1, -1, -1, -1)
                    if level_index + 1 < len(NAVIGATION_MAGNIFICATIONS):
                        child_grid = ids_by_level[NAVIGATION_MAGNIFICATIONS[level_index + 1]]
                        children = (
                            int(child_grid[row * 2, column * 2]),
                            int(child_grid[row * 2, column * 2 + 1]),
                            int(child_grid[row * 2 + 1, column * 2]),
                            int(child_grid[row * 2 + 1, column * 2 + 1]),
                        )
                    neighbors = (
                        int(identifiers[row - 1, column]) if row else -1,
                        int(identifiers[row + 1, column]) if row + 1 < rows else -1,
                        int(identifiers[row, column - 1]) if column else -1,
                        int(identifiers[row, column + 1]) if column + 1 < columns else -1,
                    )
                    embedding_index = -1
                    if magnification >= 2.5 and (row + column) % 3 != 2:
                        embedding_index = embedding_counter
                        embedding_counter += 1
                    cells[cell_id] = NavigationCell(
                        cell_id=cell_id,
                        magnification=magnification,
                        row=row,
                        column=column,
                        bounds=(
                            column * footprint,
                            row * footprint,
                            min((column + 1) * footprint, self.level0_size),
                            min((row + 1) * footprint, self.level0_size),
                        ),
                        parent_id=parent,
                        child_ids=children,
                        neighbor_ids=neighbors,
                        tissue_fraction=1.0 if (row + column) % 5 else 0.5,
                        embedding_index=embedding_index,
                    )
        coarse = ids_by_level[NAVIGATION_MAGNIFICATIONS[0]]
        start = int(coarse[coarse.shape[0] // 2, coarse.shape[1] // 2])
        return LoadedSlide(slide_id, label, overview, transform, cells, start)

    def load_slide(self, slide_id: str) -> LoadedSlide:
        return self._slides[slide_id]

    def read_viewport(self, slide_id: str, cell: NavigationCell, output_size: int) -> np.ndarray:
        label = self._slides[slide_id].label
        yy, xx = np.mgrid[:output_size, :output_size]
        return np.stack(
            (
                (xx + cell.column * 17) % 256,
                (yy + cell.row * 23) % 256,
                np.full_like(xx, int(cell.magnification * 8) + label * 40),
            ),
            axis=-1,
        ).astype(np.uint8)

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
        label = self._slides[slide_id].label
        return np.full(
            8, (1.0 if label else -1.0) / 8.0 + cell.magnification / 1000, dtype=np.float32
        )
