from __future__ import annotations

import numpy as np
import pytest

from wsi_lab.envs.observation import (
    ROIObservationTransform,
    crop_overview_to_roi,
    increment_visits,
    location_mask,
    make_observation_space,
)
from wsi_lab.wsi.reader import OverviewTransform


def test_observation_space_includes_roi_normalized_xy_scalars() -> None:
    space = make_observation_space(32, 24)

    assert space["scalars"].shape == (5,)


def test_roi_transform_round_trips_global_relative_canvas_and_normalized_coordinates() -> None:
    transform = ROIObservationTransform.fit((100, 200, 500, 400), (20, 20))

    assert transform.global_bounds == (100, 200, 500, 400)
    assert transform.content_x0 == 0
    assert transform.content_y0 == 5
    assert transform.content_width == 20
    assert transform.content_height == 10

    assert transform.global_to_roi_point(125, 230) == (25, 30)
    assert transform.roi_to_global_point(25, 30) == (125, 230)
    assert transform.global_to_roi_bounds((125, 230, 300, 350)) == (25, 30, 200, 150)
    assert transform.roi_to_global_bounds((25, 30, 200, 150)) == (125, 230, 300, 350)

    # The nominal ROI edges map exactly to zero and one, including an edge
    # that may be outside the physical WSI after grid snapping.
    assert transform.global_to_normalized_point(100, 200) == (0.0, 0.0)
    assert transform.global_to_normalized_point(500, 400) == (1.0, 1.0)
    assert transform.normalized_to_global_point(0.0, 0.0) == (100.0, 200.0)
    assert transform.normalized_to_global_point(1.0, 1.0) == (500.0, 400.0)

    assert transform.project_point(100, 200) == (0.0, 5.0)
    assert transform.project_point(500, 400) == (20.0, 15.0)
    assert transform.unproject_point(0.0, 5.0) == pytest.approx((100.0, 200.0))
    assert transform.unproject_point(20.0, 15.0) == pytest.approx((500.0, 400.0))


def test_roi_transform_rejects_empty_bounds_and_canvas() -> None:
    with pytest.raises(ValueError, match="positive area"):
        ROIObservationTransform.fit((1, 2, 1, 3), (16, 16))
    with pytest.raises(ValueError, match="canvas dimensions"):
        ROIObservationTransform.fit((1, 2, 3, 4), (0, 16))


def test_bounds_entirely_outside_roi_do_not_leak_onto_canvas_edge() -> None:
    transform = ROIObservationTransform.fit((100, 200, 500, 400), (20, 20))

    assert not location_mask(transform, (0, 0, 50, 50)).any()
    assert not location_mask(transform, (550, 450, 600, 500)).any()


def test_overview_crop_keeps_snapped_overhang_white() -> None:
    source_transform = OverviewTransform.fit((8, 4), (8, 8))
    overview = np.full((8, 8, 3), 17, dtype=np.uint8)
    # Only the actual WSI content is meaningful.  Deliberately leave the
    # source canvas padding non-white to ensure the crop uses WSI bounds.
    overview[2:6, :4] = (255, 0, 0)
    overview[2:6, 4:8] = (0, 0, 255)

    # The right and bottom halves of this snapped ROI overhang the 8x4 WSI.
    roi_transform = ROIObservationTransform.fit((4, 0, 12, 8), (8, 8))
    cropped = crop_overview_to_roi(overview, source_transform, roi_transform)

    assert cropped.shape == (8, 8, 3)
    np.testing.assert_array_equal(cropped[1, 1], np.asarray((0, 0, 255), dtype=np.uint8))
    np.testing.assert_array_equal(cropped[1, 6], np.asarray((255, 255, 255), dtype=np.uint8))
    np.testing.assert_array_equal(cropped[6, 1], np.asarray((255, 255, 255), dtype=np.uint8))
    np.testing.assert_array_equal(cropped[6, 6], np.asarray((255, 255, 255), dtype=np.uint8))
    assert np.all(cropped[:, 4:] == 255)
    assert np.all(cropped[4:, :] == 255)


def test_overview_location_and_visits_share_one_roi_letterbox_extent() -> None:
    source_transform = OverviewTransform.fit((12, 4), (12, 12))
    overview = np.full((12, 12, 3), 255, dtype=np.uint8)
    overview[4:8, :] = (20, 40, 60)
    roi_transform = ROIObservationTransform.fit((4, 0, 12, 4), (8, 8))

    cropped = crop_overview_to_roi(overview, source_transform, roi_transform)
    location = location_mask(roi_transform, roi_transform.global_bounds)
    visits = np.zeros((8, 8), dtype=np.uint32)
    increment_visits(visits, roi_transform, roi_transform.global_bounds)

    content = np.any(cropped != 255, axis=2)
    np.testing.assert_array_equal(content, location.astype(bool))
    np.testing.assert_array_equal(content, visits.astype(bool))
    assert not content[:2].any()
    assert content[2:6].all()
    assert not content[6:].any()


def test_location_and_visits_retain_legacy_full_slide_transform_support() -> None:
    transform = OverviewTransform.fit((8, 4), (8, 8))
    bounds = (2, 1, 6, 3)
    location = location_mask(transform, bounds)
    visits = np.zeros((8, 8), dtype=np.uint32)
    increment_visits(visits, transform, bounds)

    np.testing.assert_array_equal(location, visits.astype(np.float32))
