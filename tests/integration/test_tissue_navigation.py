from __future__ import annotations

import numpy as np

from wsi_lab.envs.actions import Action
from wsi_lab.envs.wsi_env import LoadedSlide, NavigationCell, WSIEnv
from wsi_lab.preprocessing.geometry import GeometryBuilder
from wsi_lab.preprocessing.pipeline import NavigationCellMapping
from wsi_lab.preprocessing.tissue import (
    TissueMask,
    apply_tissue_to_geometry,
    choose_start_cell,
)
from wsi_lab.wsi.reader import OverviewTransform


class _TopologySource:
    def __init__(self) -> None:
        geometry = GeometryBuilder(
            magnifications=(0.625, 1.25),
            footprints={0.625: 4, 1.25: 2},
        ).build(
            "islands",
            level0_width=12,
            level0_height=4,
            level0_magnification=40,
        )
        mask_values = np.zeros((4, 12), dtype=np.bool_)
        mask_values[0, 0] = True
        mask_values[0, 8] = True
        mask_transform = OverviewTransform.fit((12, 4), (12, 4))
        geometry = apply_tissue_to_geometry(
            geometry,
            TissueMask(mask_values, source="test", threshold=0.5),
            mask_transform,
        )
        cells = NavigationCellMapping(
            tuple(geometry.level(value).cells for value in geometry.magnifications)
        )
        overview_transform = OverviewTransform.fit((12, 4), (32, 32))
        self.slide = LoadedSlide(
            slide_id="islands",
            label=1,
            overview=np.zeros((32, 32, 3), dtype=np.uint8),
            overview_transform=overview_transform,
            cells=cells,
            start_cell_id=choose_start_cell(geometry),
        )

    def slide_ids(self, split: str) -> tuple[str, ...]:
        del split
        return (self.slide.slide_id,)

    def load_slide(self, slide_id: str) -> LoadedSlide:
        assert slide_id == self.slide.slide_id
        return self.slide

    def read_viewport(
        self,
        slide_id: str,
        cell: NavigationCell,
        output_size: int,
    ) -> np.ndarray:
        del slide_id, cell
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
        raise AssertionError((slide_id, cell))


def _environment() -> WSIEnv:
    return WSIEnv(
        _TopologySource(),
        overview_size=32,
        viewport_size=8,
        max_episode_steps=10,
    )


def test_precomputed_tissue_topology_drives_environment_action_mask_and_jump() -> None:
    environment = _environment()
    _, info = environment.reset(options={"slide_id": "islands"})

    assert info["column"] == 0
    assert info["action_mask"][Action.PAN_RIGHT]
    assert not info["action_mask"][Action.PAN_LEFT]

    _, _, _, _, info = environment.step(Action.PAN_RIGHT)

    assert info["column"] == 2
    assert info["action_mask"][Action.PAN_LEFT]
    assert not info["action_mask"][Action.PAN_RIGHT]


def test_precomputed_child_validity_masks_zoom_without_using_embeddings() -> None:
    environment = _environment()
    _, info = environment.reset(options={"slide_id": "islands"})

    assert info["action_mask"][Action.ZOOM_IN_TL]
    assert not info["action_mask"][Action.ZOOM_IN_TR]
    assert not info["action_mask"][Action.ZOOM_IN_BL]
    assert not info["action_mask"][Action.ZOOM_IN_BR]

    _, _, _, _, info = environment.step(Action.ZOOM_IN_TL)

    assert info["magnification"] == 1.25
    assert info["evidence_count"] == 0
    assert info["action_mask"][Action.ZOOM_OUT]
