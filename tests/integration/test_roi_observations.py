from __future__ import annotations

import numpy as np
import pytest

from wsi_lab.envs.actions import Action
from wsi_lab.envs.observation import ROIObservationTransform
from wsi_lab.envs.wsi_env import LoadedSlide, NavigationCell, WSIEnv
from wsi_lab.wsi.reader import OverviewTransform


class _OffsetIslandsSource:
    """One offset ROI containing two tissue islands separated by background."""

    def __init__(self) -> None:
        full_transform = OverviewTransform.fit((100, 80), (40, 40))
        navigation_roi = (20, 20, 80, 40)
        overview = np.full((40, 40, 3), 255, dtype=np.uint8)
        overview[4:36] = (0, 0, 255)
        roi_x0, roi_y0, roi_x1, roi_y1 = full_transform.raster_bounds(
            navigation_roi,
            ensure_nonempty=False,
        )
        overview[roi_y0:roi_y1, roi_x0:roi_x1] = (255, 0, 0)

        cells = {
            0: NavigationCell(
                cell_id=0,
                magnification=0.625,
                row=0,
                column=0,
                bounds=(20, 20, 40, 40),
                child_ids=(10, -1, -1, -1),
                # Cell 1 is background; panning jumps directly to tissue cell 2.
                neighbor_ids=(-1, -1, -1, 2),
                tissue_fraction=1.0,
                tissue_valid=True,
            ),
            1: NavigationCell(
                cell_id=1,
                magnification=0.625,
                row=0,
                column=1,
                bounds=(40, 20, 60, 40),
                tissue_fraction=0.0,
                tissue_valid=False,
            ),
            2: NavigationCell(
                cell_id=2,
                magnification=0.625,
                row=0,
                column=2,
                bounds=(60, 20, 80, 40),
                neighbor_ids=(-1, -1, 0, -1),
                tissue_fraction=1.0,
                tissue_valid=True,
            ),
            10: NavigationCell(
                cell_id=10,
                magnification=1.25,
                row=0,
                column=0,
                bounds=(20, 20, 30, 30),
                parent_id=0,
                child_ids=(20, -1, -1, -1),
                tissue_fraction=1.0,
                tissue_valid=True,
            ),
            20: NavigationCell(
                cell_id=20,
                magnification=2.5,
                row=0,
                column=0,
                bounds=(20, 20, 25, 25),
                parent_id=10,
                tissue_fraction=1.0,
                embedding_index=7,
                tissue_valid=True,
            ),
        }
        self.slide = LoadedSlide(
            slide_id="offset_islands",
            label=1,
            overview=overview,
            overview_transform=full_transform,
            cells=cells,
            start_cell_id=0,
            navigation_roi_level0=navigation_roi,
        )
        self.viewport_requests: list[NavigationCell] = []
        self.embedding_requests: list[NavigationCell] = []

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
        assert slide_id == self.slide.slide_id
        self.viewport_requests.append(cell)
        pixel = np.asarray((cell.bounds[0], cell.bounds[1], cell.bounds[2]), dtype=np.uint8)
        return np.broadcast_to(pixel, (output_size, output_size, 3)).copy()

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
        assert slide_id == self.slide.slide_id
        self.embedding_requests.append(cell)
        return np.full(8, float(cell.embedding_index), dtype=np.float32)


def _environment() -> tuple[WSIEnv, _OffsetIslandsSource]:
    source = _OffsetIslandsSource()
    return (
        WSIEnv(
            source,
            overview_size=40,
            viewport_size=8,
            max_episode_steps=20,
        ),
        source,
    )


def test_environment_uses_one_offset_roi_transform_for_all_spatial_channels() -> None:
    environment, source = _environment()
    observation, _ = environment.reset(options={"slide_id": source.slide.slide_id})

    transform = environment.observation_transform
    assert isinstance(transform, ROIObservationTransform)
    assert transform.global_bounds == (20, 20, 80, 40)
    assert transform.global_to_roi_point(50, 30) == (30, 10)
    assert transform.roi_to_global_point(30, 10) == (50, 30)
    assert transform.unproject_point(*transform.project_point(70, 30)) == pytest.approx((70, 30))

    # Both disconnected tissue islands lie in the same nominal spatial world.
    for cell_id in (0, 2):
        normalized = transform.global_to_normalized_bounds(source.slide.cells[cell_id].bounds)
        assert min(normalized) >= 0.0
        assert max(normalized) <= 1.0

    full_extent = np.zeros((40, 40), dtype=np.bool_)
    x0, y0, x1, y1 = transform.raster_bounds(transform.global_bounds)
    full_extent[y0:y1, x0:x1] = True
    overview_content = np.any(observation["global"][:3] < 1.0, axis=0)
    np.testing.assert_array_equal(overview_content, full_extent)
    np.testing.assert_allclose(
        observation["global"][:3, (y0 + y1) // 2, (x0 + x1) // 2],
        (1.0, 0.0, 0.0),
        atol=1.0 / 255.0,
    )

    expected_location = np.zeros((40, 40), dtype=np.float32)
    x0, y0, x1, y1 = transform.raster_bounds(source.slide.cells[0].bounds)
    expected_location[y0:y1, x0:x1] = 1.0
    np.testing.assert_array_equal(observation["global"][3], expected_location)
    np.testing.assert_array_equal(observation["global"][4] > 0, expected_location.astype(bool))
    np.testing.assert_allclose(observation["global"][4], expected_location * 0.1)

    environment.close()


def test_normalized_position_uses_roi_edges_and_current_global_cell_center() -> None:
    environment, source = _environment()
    observation, _ = environment.reset(options={"slide_id": source.slide.slide_id})
    transform = environment.observation_transform

    assert transform.global_to_normalized_point(20, 20) == (0.0, 0.0)
    assert transform.global_to_normalized_point(80, 40) == (1.0, 1.0)
    assert transform.normalized_to_global_point(0.0, 0.0) == (20.0, 20.0)
    assert transform.normalized_to_global_point(1.0, 1.0) == (80.0, 40.0)
    np.testing.assert_allclose(observation["scalars"][3:], (1.0 / 6.0, 0.5))

    observation, _, _, _, info = environment.step(Action.PAN_RIGHT)

    assert info["cell_id"] == 2
    np.testing.assert_allclose(observation["scalars"][3:], (5.0 / 6.0, 0.5))
    environment.close()


def test_roi_observations_do_not_change_global_navigation_viewport_or_embedding_lookup() -> None:
    environment, source = _environment()
    observation, info = environment.reset(options={"slide_id": source.slide.slide_id})

    assert not info["action_mask"][Action.PAN_UP]
    assert not info["action_mask"][Action.PAN_DOWN]
    assert not info["action_mask"][Action.PAN_LEFT]
    assert info["action_mask"][Action.PAN_RIGHT]
    assert info["action_mask"][Action.ZOOM_IN_TL]
    assert source.viewport_requests[-1] is source.slide.cells[0]
    np.testing.assert_allclose(
        observation["viewport"][:, 0, 0],
        np.asarray((20, 20, 40), dtype=np.float32) / 255.0,
        atol=0,
    )

    observation, _, _, _, info = environment.step(Action.PAN_RIGHT)
    assert info["cell_id"] == 2
    assert source.viewport_requests[-1] is source.slide.cells[2]
    assert source.viewport_requests[-1].bounds == (60, 20, 80, 40)
    assert info["action_mask"][Action.PAN_LEFT]
    assert not info["action_mask"][Action.PAN_RIGHT]
    np.testing.assert_allclose(
        observation["viewport"][:, 0, 0],
        np.asarray((60, 20, 80), dtype=np.float32) / 255.0,
        atol=0,
    )

    environment.reset(options={"slide_id": source.slide.slide_id})
    environment.step(Action.ZOOM_IN_TL)
    _, _, _, _, info = environment.step(Action.ZOOM_IN_TL)

    assert info["cell_id"] == 20
    assert info["evidence_added"]
    assert source.embedding_requests == [source.slide.cells[20]]
    embedded_cell = source.embedding_requests[0]
    assert embedded_cell is source.slide.cells[20]
    assert embedded_cell.bounds == (20, 20, 25, 25)
    assert embedded_cell.embedding_index == 7
    assert source.viewport_requests[-1] is source.slide.cells[20]
    assert source.slide.cells[0].neighbor_ids[Action.PAN_RIGHT] == 2

    environment.close()
