from __future__ import annotations

import numpy as np
import pytest

from wsi_lab.envs.actions import Action
from wsi_lab.envs.augmentation import GeometricOrientation, HEStainAugmentation

ORIENTATIONS = tuple(
    GeometricOrientation(quarter_turns, horizontal_flip)
    for horizontal_flip in (False, True)
    for quarter_turns in range(4)
)


def test_all_eight_image_orientations_are_unique_and_invertible() -> None:
    image = np.arange(2 * 3 * 2, dtype=np.uint8).reshape(2, 3, 2)

    transformed = [orientation.apply_image(image) for orientation in ORIENTATIONS]

    assert len({(result.shape, result.tobytes()) for result in transformed}) == 8
    for orientation, result in zip(ORIENTATIONS, transformed, strict=True):
        np.testing.assert_array_equal(orientation.inverse_image(result), image)
        assert result.flags.c_contiguous


def test_rotation_is_clockwise_and_horizontal_flip_is_applied_afterward() -> None:
    image = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)

    rotated = GeometricOrientation(1).apply_image(image)
    rotated_and_flipped = GeometricOrientation(1, True).apply_image(image)

    np.testing.assert_array_equal(rotated, [[3, 0], [4, 1], [5, 2]])
    np.testing.assert_array_equal(rotated_and_flipped, [[0, 3], [1, 4], [2, 5]])


@pytest.mark.parametrize("orientation", ORIENTATIONS, ids=lambda value: value.name)
def test_points_and_bounds_round_trip_on_a_rectangular_padded_frame(
    orientation: GeometricOrientation,
) -> None:
    width, height = 11, 7
    bounds = (1, 2, 8, 6)

    for point in ((0.0, 0.0), (1.25, 2.5), (11.0, 7.0)):
        displayed = orientation.transform_point(*point, width, height)
        assert orientation.inverse_point(*displayed, width, height) == pytest.approx(point)
    assert (
        orientation.inverse_bounds(
            orientation.transform_bounds(bounds, width, height), width, height
        )
        == bounds
    )


def test_rectangular_frame_point_bounds_and_size_have_expected_coordinates() -> None:
    orientation = GeometricOrientation(1, True)

    assert orientation.transformed_size(6, 4) == (4, 6)
    assert orientation.transform_point(1, 1, 6, 4) == (1.0, 1.0)
    assert orientation.transform_bounds((1, 1, 3, 2), 6, 4) == (1, 1, 2, 3)


@pytest.mark.parametrize("orientation", ORIENTATIONS, ids=lambda value: value.name)
def test_grid_rows_and_columns_match_image_orientation(
    orientation: GeometricOrientation,
) -> None:
    rows, columns = 2, 3
    canonical = np.arange(rows * columns).reshape(rows, columns)
    displayed = orientation.apply_image(canonical)

    assert orientation.transformed_grid_shape(rows, columns) == displayed.shape
    for row in range(rows):
        for column in range(columns):
            displayed_index = orientation.transform_row_column(row, column, rows, columns)
            assert displayed[displayed_index] == canonical[row, column]
            assert orientation.inverse_row_column(*displayed_index, rows, columns) == (row, column)


@pytest.mark.parametrize("orientation", ORIENTATIONS, ids=lambda value: value.name)
def test_agent_and_canonical_actions_are_inverse_permutations(
    orientation: GeometricOrientation,
) -> None:
    for action in Action:
        displayed = orientation.from_canonical_action(action)
        assert orientation.to_canonical_action(displayed) is action

    for invariant in (
        Action.ZOOM_OUT,
        Action.DIAGNOSE_CANCEROUS,
        Action.DIAGNOSE_BENIGN,
    ):
        assert orientation.from_canonical_action(invariant) is invariant
        assert orientation.to_canonical_action(invariant) is invariant


def test_actions_follow_the_pixels_for_clockwise_rotation() -> None:
    orientation = GeometricOrientation(1)

    assert orientation.from_canonical_action(Action.PAN_UP) is Action.PAN_RIGHT
    assert orientation.to_canonical_action(Action.PAN_RIGHT) is Action.PAN_UP
    assert orientation.from_canonical_action(Action.ZOOM_IN_TL) is Action.ZOOM_IN_TR
    assert orientation.to_canonical_action(Action.ZOOM_IN_TR) is Action.ZOOM_IN_TL


@pytest.mark.parametrize("orientation", ORIENTATIONS, ids=lambda value: value.name)
def test_action_mask_is_permuted_from_canonical_semantics(
    orientation: GeometricOrientation,
) -> None:
    canonical = np.arange(len(Action), dtype=np.int64)

    displayed = orientation.mask_from_canonical(canonical)

    assert displayed.flags.c_contiguous
    for action in Action:
        assert displayed[action] == canonical[orientation.to_canonical_action(action)]


def test_geometric_sampling_is_seeded_and_can_reach_every_orientation() -> None:
    first_rng = np.random.default_rng(71)
    second_rng = np.random.default_rng(71)

    first = [GeometricOrientation.sample(first_rng) for _ in range(100)]
    second = [GeometricOrientation.sample(second_rng) for _ in range(100)]

    assert first == second
    assert set(first) == set(ORIENTATIONS)


def test_stain_sampling_is_seeded_independent_and_in_the_mild_range() -> None:
    first = HEStainAugmentation.sample(np.random.default_rng(19))
    second = HEStainAugmentation.sample(np.random.default_rng(19))

    assert first == second
    assert 0.9 <= first.hematoxylin_scale <= 1.1
    assert 0.9 <= first.eosin_scale <= 1.1
    assert first.hematoxylin_scale != first.eosin_scale


def test_stain_augmentation_is_episode_consistent_and_preserves_white_padding() -> None:
    image = np.asarray(
        [
            [[255, 255, 255], [120, 80, 140], [220, 160, 190]],
            [[60, 40, 90], [245, 245, 245], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    augmentation = HEStainAugmentation(hematoxylin_scale=1.1, eosin_scale=0.9)

    first = augmentation.apply_image(image)
    second = augmentation.apply_image(image)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(first[1, 2], [255, 255, 255])
    assert np.any(first[~np.all(image == 255, axis=-1)] != image[~np.all(image == 255, axis=-1)])
    assert first.dtype == np.uint8
    assert first.shape == image.shape
    assert first.flags.c_contiguous


def test_unit_stain_scales_are_byte_exact_and_do_not_alias_input() -> None:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    result = HEStainAugmentation().apply_image(image)

    np.testing.assert_array_equal(result, image)
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, image)


def test_stain_transform_keeps_residual_optical_density_unchanged() -> None:
    image = np.asarray([[[96, 54, 121], [160, 92, 144]]], dtype=np.uint8)
    result = HEStainAugmentation(1.08, 0.92).apply_image(image)
    hematoxylin = np.asarray((0.650, 0.704, 0.286), dtype=np.float64)
    eosin = np.asarray((0.072, 0.990, 0.105), dtype=np.float64)
    hematoxylin /= np.linalg.norm(hematoxylin)
    eosin /= np.linalg.norm(eosin)
    residual = np.cross(hematoxylin, eosin)
    residual /= np.linalg.norm(residual)
    inverse_basis = np.linalg.inv(np.column_stack((hematoxylin, eosin, residual)))

    def residual_concentration(rgb: np.ndarray) -> np.ndarray:
        optical_density = -np.log((rgb.astype(np.float64) + 1.0) / 256.0)
        return (optical_density.reshape(-1, 3) @ inverse_basis.T)[:, 2]

    np.testing.assert_allclose(
        residual_concentration(result),
        residual_concentration(image),
        atol=0.015,
    )


def test_augmentation_rejects_invalid_shapes_and_parameters() -> None:
    with pytest.raises(ValueError, match="quarter_turns"):
        GeometricOrientation(4)
    with pytest.raises(ValueError, match="at least two"):
        GeometricOrientation().apply_image(np.zeros(3, dtype=np.uint8))
    with pytest.raises(ValueError, match="action mask"):
        GeometricOrientation().mask_from_canonical(np.ones(10, dtype=bool))
    with pytest.raises(ValueError, match="stain scales"):
        HEStainAugmentation(0.0, 1.0)
    with pytest.raises(ValueError, match="shape"):
        HEStainAugmentation(1.1, 0.9).apply_image(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        HEStainAugmentation(1.1, 0.9).apply_image(np.zeros((4, 4, 3), dtype=np.float32))
