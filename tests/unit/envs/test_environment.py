from __future__ import annotations

import numpy as np
import pytest

from wsi_lab.constants import LEVEL0_FOOTPRINTS
from wsi_lab.envs.actions import Action
from wsi_lab.envs.augmentation import GeometricOrientation, HEStainAugmentation
from wsi_lab.envs.wsi_env import LoadedSlide, NavigationCell, make_env
from wsi_lab.testing.synthetic import DeterministicBelief, SyntheticEpisodeSource
from wsi_lab.wsi.reader import OverviewTransform


class TissueGraphSource:
    """Small graph whose right pan jumps over a disconnected background cell."""

    def __init__(self) -> None:
        overview = np.zeros((32, 32, 3), dtype=np.uint8)
        transform = OverviewTransform.fit((49152, 16384), (32, 32))
        cells = {
            0: NavigationCell(
                cell_id=0,
                magnification=0.625,
                row=0,
                column=0,
                bounds=(0, 0, 16384, 16384),
                child_ids=(10, 11, -1, -1),
                # ID 2 is the next tissue location; ID 1 is background.
                neighbor_ids=(-1, -1, -1, 2),
                tissue_fraction=1.0,
                tissue_valid=True,
            ),
            1: NavigationCell(
                cell_id=1,
                magnification=0.625,
                row=0,
                column=1,
                bounds=(16384, 0, 32768, 16384),
                tissue_fraction=0.0,
                tissue_valid=False,
            ),
            2: NavigationCell(
                cell_id=2,
                magnification=0.625,
                row=0,
                column=2,
                bounds=(32768, 0, 49152, 16384),
                neighbor_ids=(-1, -1, 0, -1),
                tissue_fraction=1.0,
                tissue_valid=True,
            ),
            10: NavigationCell(
                cell_id=10,
                magnification=1.25,
                row=0,
                column=0,
                bounds=(0, 0, 8192, 8192),
                parent_id=0,
                tissue_fraction=0.4,
                embedding_index=-1,
                tissue_valid=True,
            ),
            11: NavigationCell(
                cell_id=11,
                magnification=1.25,
                row=0,
                column=1,
                bounds=(8192, 0, 16384, 8192),
                parent_id=0,
                tissue_fraction=0.0,
                embedding_index=-1,
                tissue_valid=False,
            ),
        }
        self.slide = LoadedSlide("tissue_graph", 1, overview, transform, cells, 0)

    def slide_ids(self, split: str):
        del split
        return (self.slide.slide_id,)

    def load_slide(self, slide_id: str) -> LoadedSlide:
        assert slide_id == self.slide.slide_id
        return self.slide

    def read_viewport(self, slide_id: str, cell: NavigationCell, output_size: int) -> np.ndarray:
        del slide_id, cell
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
        raise AssertionError((slide_id, cell))


class RasterRoundingSource:
    """Offset rectangular ROI whose fine cells expose raster rounding drift."""

    def __init__(self) -> None:
        coarse = LEVEL0_FOOTPRINTS[0.625]
        width, height = 42 * coarse, 13 * coarse
        overview = np.zeros((224, 224, 3), dtype=np.uint8)
        yy, xx = np.mgrid[:224, :224]
        overview[..., 0] = xx
        overview[..., 1] = yy
        cells = {
            0: NavigationCell(
                cell_id=0,
                magnification=0.625,
                row=7,
                column=5,
                bounds=(5 * coarse, 7 * coarse, 6 * coarse, 8 * coarse),
                child_ids=(1, -1, -1, -1),
                tissue_valid=True,
            ),
            1: NavigationCell(
                cell_id=1,
                magnification=1.25,
                row=14,
                column=10,
                bounds=(5 * coarse, 7 * coarse, 11 * coarse // 2, 15 * coarse // 2),
                parent_id=0,
                child_ids=(-1, -1, 2, -1),
                tissue_valid=True,
            ),
            2: NavigationCell(
                cell_id=2,
                magnification=2.5,
                row=29,
                column=20,
                bounds=(5 * coarse, 29 * coarse // 4, 21 * coarse // 4, 15 * coarse // 2),
                parent_id=1,
                tissue_valid=True,
            ),
        }
        self.slide = LoadedSlide(
            "raster_rounding",
            0,
            overview,
            OverviewTransform.fit((width, height), (224, 224)),
            cells,
            0,
            navigation_roi_level0=(4 * coarse, 5 * coarse, 16 * coarse, 12 * coarse),
        )

    def slide_ids(self, split: str):
        del split
        return (self.slide.slide_id,)

    def load_slide(self, slide_id: str) -> LoadedSlide:
        assert slide_id == self.slide.slide_id
        return self.slide

    def read_viewport(self, slide_id: str, cell: NavigationCell, output_size: int) -> np.ndarray:
        del slide_id, cell
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
        raise AssertionError((slide_id, cell))


def make_test_env(**kwargs):
    return make_env(
        source=SyntheticEpisodeSource(overview_size=32),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
        **kwargs,
    )


def test_environment_defaults_to_32_diagnostic_embeddings() -> None:
    env = make_test_env()

    assert env.min_diagnostic_evidence == 32
    env.close()


def test_start_position_defaults_to_preprocessed_center() -> None:
    source = SyntheticEpisodeSource(overview_size=32)
    env = make_env(source=source, overview_size=32, viewport_size=24)

    _, info = env.reset(seed=7, options={"slide_id": "synthetic_cancer"})

    assert info["cell_id"] == source.load_slide("synthetic_cancer").start_cell_id
    assert info["start_position"] == "center"
    env.close()


def test_random_start_position_is_seeded_and_uses_coarsest_tissue_cells() -> None:
    source = SyntheticEpisodeSource(overview_size=32)
    env = make_env(
        source=source,
        overview_size=32,
        viewport_size=24,
        start_position="random",
    )
    slide = source.load_slide("synthetic_cancer")
    valid_ids = {
        cell.cell_id
        for cell in slide.cells.values()
        if cell.magnification == 0.625 and cell.tissue_valid
    }

    _, first = env.reset(seed=11, options={"slide_id": slide.slide_id})
    _, repeated = env.reset(seed=11, options={"slide_id": slide.slide_id})
    sampled = {
        env.reset(seed=seed, options={"slide_id": slide.slide_id})[1]["cell_id"]
        for seed in range(20)
    }

    assert first["cell_id"] == repeated["cell_id"]
    assert first["start_position"] == "random"
    assert sampled <= valid_ids
    assert len(sampled) > 1
    env.close()


def test_random_start_position_excludes_background_cells() -> None:
    source = TissueGraphSource()
    env = make_env(
        source=source,
        overview_size=32,
        viewport_size=24,
        start_position="random",
    )

    sampled = {
        env.reset(seed=seed, options={"slide_id": source.slide.slide_id})[1]["cell_id"]
        for seed in range(20)
    }

    assert sampled == {0, 2}
    env.close()


def test_start_position_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="start_position"):
        make_test_env(start_position="edge")


def _reset_with_orientation(env, name: str, *, slide_id: str):
    for seed in range(256):
        observation, info = env.reset(seed=seed, options={"slide_id": slide_id})
        if info["geometric_orientation"] == name:
            return seed, observation, info
    raise AssertionError(f"orientation {name!r} was not sampled")


def test_augmentation_defaults_are_identity_and_reported() -> None:
    env = make_test_env()

    _, info = env.reset(seed=5, options={"slide_id": "synthetic_cancer"})

    assert info["geometric_augmentation"] is False
    assert info["geometric_orientation"] == "rot0"
    assert info["photometric_augmentation"] is False
    assert info["hematoxylin_scale"] == pytest.approx(1.0)
    assert info["eosin_scale"] == pytest.approx(1.0)
    assert info["cell_bounds_level0"] == info["canonical_cell_bounds_level0"]
    assert info["navigation_roi_level0"] == info["canonical_navigation_roi_level0"]
    env.close()


def test_augmentation_rng_does_not_change_slide_or_random_start_sequence() -> None:
    plain = make_test_env(start_position="random")
    augmented = make_test_env(
        start_position="random",
        geometric_augmentation=True,
        photometric_augmentation=True,
    )

    plain_sequence = []
    augmented_sequence = []
    for episode in range(12):
        seed = 41 if episode == 0 else None
        _, plain_info = plain.reset(seed=seed)
        _, augmented_info = augmented.reset(seed=seed)
        plain_sequence.append((plain_info["slide_id"], plain_info["cell_id"]))
        augmented_sequence.append((augmented_info["slide_id"], augmented_info["cell_id"]))

    assert augmented_sequence == plain_sequence
    plain.close()
    augmented.close()


def test_geometric_environment_rotates_spatial_observations_and_actions() -> None:
    orientation = GeometricOrientation(1)
    source = TissueGraphSource()
    env = make_env(
        source=source,
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
        geometric_augmentation=True,
    )
    seed, observation, info = _reset_with_orientation(
        env,
        orientation.name,
        slide_id=source.slide.slide_id,
    )

    assert info["action_mask"][Action.PAN_DOWN]
    assert not info["action_mask"][Action.PAN_RIGHT]
    assert info["action_mask"][Action.ZOOM_IN_TR]
    assert not info["action_mask"][Action.ZOOM_IN_TL]
    assert info["cell_bounds_level0"] == (0, 0, 16384, 16384)
    assert info["canonical_cell_bounds_level0"] == (0, 0, 16384, 16384)
    assert info["augmentation_frame_size_level0"] == (16384, 49152)
    assert observation["scalars"][3:].tolist() == pytest.approx([0.5, 1 / 6])

    _, _, _, _, moved = env.step(Action.PAN_DOWN)

    assert moved["cell_id"] == 2
    assert (moved["row"], moved["column"]) == (2, 0)
    assert (moved["canonical_row"], moved["canonical_column"]) == (0, 2)
    assert moved["cell_bounds_level0"] == (0, 32768, 16384, 49152)
    assert env.trajectory[-1]["action"] == int(Action.PAN_DOWN)
    assert env.trajectory[-1]["canonical_bounds"] == [32768, 0, 49152, 16384]
    assert env.render().shape == (32, 32, 3)

    env.reset(seed=seed, options={"slide_id": source.slide.slide_id})
    _, _, _, _, zoomed = env.step(Action.ZOOM_IN_TR)
    assert zoomed["cell_id"] == 10
    env.close()


def test_geometric_observation_is_the_orientation_of_the_canonical_observation() -> None:
    orientation = GeometricOrientation(1, True)
    plain = make_test_env()
    augmented = make_test_env(geometric_augmentation=True)
    seed, transformed, info = _reset_with_orientation(
        augmented,
        orientation.name,
        slide_id="synthetic_cancer",
    )
    canonical, canonical_info = plain.reset(
        seed=seed,
        options={"slide_id": "synthetic_cancer"},
    )

    expected_global = np.moveaxis(
        orientation.apply_image(np.moveaxis(canonical["global"], 0, -1)),
        -1,
        0,
    )
    expected_viewport = np.moveaxis(
        orientation.apply_image(np.moveaxis(canonical["viewport"], 0, -1)),
        -1,
        0,
    )
    np.testing.assert_array_equal(transformed["global"], expected_global)
    np.testing.assert_array_equal(transformed["viewport"], expected_viewport)
    np.testing.assert_array_equal(
        info["action_mask"],
        orientation.mask_from_canonical(canonical_info["action_mask"]),
    )
    assert augmented.observation_space.contains(transformed)
    plain.close()
    augmented.close()


def test_geometric_channels_rotate_canonical_rasters_without_rounding_drift() -> None:
    orientation = GeometricOrientation(1)
    source = RasterRoundingSource()
    plain = make_env(source=source, overview_size=224, viewport_size=24)
    augmented = make_env(
        source=source,
        overview_size=224,
        viewport_size=24,
        geometric_augmentation=True,
    )
    seed, transformed, _ = _reset_with_orientation(
        augmented,
        orientation.name,
        slide_id=source.slide.slide_id,
    )
    canonical, _ = plain.reset(seed=seed, options={"slide_id": source.slide.slide_id})

    for canonical_action in (Action.ZOOM_IN_TL, Action.ZOOM_IN_BL):
        canonical, _, _, _, _ = plain.step(canonical_action)
        transformed, _, _, _, _ = augmented.step(
            orientation.from_canonical_action(canonical_action)
        )

    expected = np.moveaxis(
        orientation.apply_image(np.moveaxis(canonical["global"], 0, -1)),
        -1,
        0,
    )
    np.testing.assert_array_equal(transformed["global"], expected)
    np.testing.assert_array_equal(augmented.render(), orientation.apply_image(plain.render()))
    plain.close()
    augmented.close()


def test_photometric_environment_reuses_one_seeded_stain_transform_per_episode() -> None:
    plain = make_test_env()
    augmented = make_test_env(photometric_augmentation=True)
    canonical, _ = plain.reset(seed=23, options={"slide_id": "synthetic_cancer"})
    transformed, info = augmented.reset(seed=23, options={"slide_id": "synthetic_cancer"})
    repeated, repeated_info = augmented.reset(
        seed=23,
        options={"slide_id": "synthetic_cancer"},
    )
    stain = HEStainAugmentation(info["hematoxylin_scale"], info["eosin_scale"])
    overview_uint8 = np.moveaxis(np.rint(canonical["global"][:3] * 255), 0, -1).astype(np.uint8)
    viewport_uint8 = np.moveaxis(np.rint(canonical["viewport"] * 255), 0, -1).astype(np.uint8)

    expected_overview = np.moveaxis(stain.apply_image(overview_uint8), -1, 0) / 255.0
    expected_viewport = np.moveaxis(stain.apply_image(viewport_uint8), -1, 0) / 255.0
    np.testing.assert_allclose(transformed["global"][:3], expected_overview, atol=1e-7)
    np.testing.assert_allclose(transformed["viewport"], expected_viewport, atol=1e-7)
    np.testing.assert_array_equal(transformed["global"][3:], canonical["global"][3:])
    np.testing.assert_array_equal(repeated["global"], transformed["global"])
    np.testing.assert_array_equal(repeated["viewport"], transformed["viewport"])
    assert repeated_info["hematoxylin_scale"] == info["hematoxylin_scale"]
    assert repeated_info["eosin_scale"] == info["eosin_scale"]
    assert 0.9 <= info["hematoxylin_scale"] <= 1.1
    assert 0.9 <= info["eosin_scale"] <= 1.1
    assert augmented.observation_space.contains(transformed)
    plain.close()
    augmented.close()


def test_geometric_navigation_keeps_embedding_lookup_on_the_canonical_cell() -> None:
    class TrackingSource(SyntheticEpisodeSource):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.embedding_cells: list[tuple[int, int]] = []

        def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
            self.embedding_cells.append((cell.cell_id, cell.embedding_index))
            return super().read_embedding(slide_id, cell)

    source = TrackingSource(overview_size=32)
    env = make_env(
        source=source,
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
        geometric_augmentation=True,
    )
    orientation = GeometricOrientation(3, True)
    _, _, _ = _reset_with_orientation(
        env,
        orientation.name,
        slide_id="synthetic_cancer",
    )
    for canonical_action in (
        Action.PAN_UP,
        Action.PAN_LEFT,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
    ):
        _, _, _, _, info = env.step(orientation.from_canonical_action(canonical_action))

    assert info["evidence_added"]
    assert info["evidence_count"] == 1
    assert source.embedding_cells == [
        (
            info["cell_id"],
            source.load_slide("synthetic_cancer").cells[info["cell_id"]].embedding_index,
        )
    ]
    env.close()


def advance_to_first_evidence(env):
    info = None
    reward = 0.0
    for action in (
        Action.PAN_UP,
        Action.PAN_LEFT,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
    ):
        _, reward, _, _, info = env.step(action)
    assert info is not None
    return reward, info


def test_observation_contract_and_initial_visit() -> None:
    env = make_test_env()
    observation, info = env.reset(seed=2, options={"slide_id": "synthetic_cancer"})
    assert observation["global"].shape == (5, 32, 32)
    assert observation["viewport"].shape == (3, 24, 24)
    assert observation["scalars"].tolist() == pytest.approx([0.0, 1.0, 0.5, 0.75, 0.75])
    assert observation["global"][4].max() == pytest.approx(0.1)
    assert info["evidence_count"] == 0
    assert info["action_mask"].dtype == np.bool_
    env.close()


def test_lightweight_evaluation_preserves_navigation_state_without_rgb_work() -> None:
    class CountingSource(SyntheticEpisodeSource):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.viewport_reads = 0

        def read_viewport(
            self, slide_id: str, cell: NavigationCell, output_size: int
        ) -> np.ndarray:
            self.viewport_reads += 1
            return super().read_viewport(slide_id, cell, output_size)

    standard_source = CountingSource(overview_size=32)
    lightweight_source = CountingSource(overview_size=32)
    standard = make_env(
        source=standard_source,
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
        max_episode_steps=20,
    )
    lightweight = make_env(
        source=lightweight_source,
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
        max_episode_steps=20,
        lightweight_evaluation=True,
    )

    standard_observation, standard_info = standard.reset(
        seed=3, options={"slide_id": "synthetic_cancer"}
    )
    lightweight_observation, lightweight_info = lightweight.reset(
        seed=3, options={"slide_id": "synthetic_cancer"}
    )
    assert set(lightweight_observation) == {"scalars"}
    np.testing.assert_allclose(lightweight_observation["scalars"], standard_observation["scalars"])
    np.testing.assert_array_equal(
        lightweight_info["action_mask"], standard_info["action_mask"]
    )

    for action in (
        Action.PAN_UP,
        Action.PAN_LEFT,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
    ):
        standard_observation, standard_reward, _, _, standard_info = standard.step(action)
        lightweight_observation, lightweight_reward, _, _, lightweight_info = lightweight.step(
            action
        )
        assert lightweight_reward == pytest.approx(standard_reward)
        np.testing.assert_allclose(
            lightweight_observation["scalars"], standard_observation["scalars"]
        )
        np.testing.assert_array_equal(
            lightweight_info["action_mask"], standard_info["action_mask"]
        )
        for field in (
            "cell_id",
            "belief",
            "unique_evidence_count",
            "actions_taken",
            "revisit_rate",
            "inspected_fraction",
            "magnification_histogram",
        ):
            assert lightweight_info[field] == standard_info[field]

    assert standard_source.viewport_reads == 6
    assert lightweight_source.viewport_reads == 0
    assert lightweight.unwrapped.trajectory == ()
    with pytest.raises(RuntimeError, match="lightweight evaluation"):
        lightweight.render()
    standard.close()
    lightweight.close()


def test_invalid_action_is_noop_without_budget_or_visit() -> None:
    env = make_test_env()
    before, info = env.reset(seed=1, options={"slide_id": "synthetic_benign"})
    assert not info["action_mask"][Action.PAN_DOWN]
    after, reward, terminated, truncated, next_info = env.step(Action.PAN_DOWN)
    assert reward == pytest.approx(-0.001)
    assert not terminated and not truncated
    assert next_info["invalid_action"]
    assert next_info["actions_taken"] == 0
    np.testing.assert_array_equal(after["global"], before["global"])


def test_sparse_embedding_child_remains_navigable_without_belief_update() -> None:
    env = make_test_env(min_diagnostic_evidence=1)
    _, info = env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    for action in (
        Action.PAN_UP,
        Action.PAN_LEFT,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
    ):
        _, _, _, _, info = env.step(action)
    assert info["magnification"] == 5.0
    before_belief = info["belief"]
    before_count = info["evidence_count"]
    assert info["action_mask"][Action.ZOOM_IN_BR]
    _, _, _, _, info = env.step(Action.ZOOM_IN_BR)
    assert info["magnification"] == 10.0
    assert info["evidence_count"] == before_count
    assert info["belief"] == before_belief


def test_tissue_graph_masks_background_and_jumps_to_disconnected_tissue() -> None:
    env = make_env(
        source=TissueGraphSource(),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
    )
    _, info = env.reset(options={"slide_id": "tissue_graph"})

    assert not info["action_mask"][Action.PAN_UP]
    assert not info["action_mask"][Action.PAN_DOWN]
    assert not info["action_mask"][Action.PAN_LEFT]
    assert info["action_mask"][Action.PAN_RIGHT]

    _, _, _, _, info = env.step(Action.PAN_RIGHT)
    assert info["cell_id"] == 2
    assert info["column"] == 2


def test_zoom_masks_non_tissue_child_but_missing_embedding_tissue_is_navigable() -> None:
    env = make_env(
        source=TissueGraphSource(),
        belief=DeterministicBelief(),
        overview_size=32,
        viewport_size=24,
    )
    _, info = env.reset(options={"slide_id": "tissue_graph"})

    assert info["action_mask"][Action.ZOOM_IN_TL]
    assert not info["action_mask"][Action.ZOOM_IN_TR]
    _, reward, _, _, info = env.step(Action.ZOOM_IN_TL)
    assert info["cell_id"] == 10
    assert info["evidence_count"] == 0
    assert reward == pytest.approx(-0.001)
    assert info["action_mask"][Action.ZOOM_OUT]


def test_final_action_is_reserved_for_diagnosis() -> None:
    env = make_test_env(max_episode_steps=2, min_diagnostic_evidence=99)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    _, _, terminated, _, info = env.step(Action.PAN_UP)
    assert not terminated
    assert not info["action_mask"][:9].any()
    assert info["action_mask"][9:].all()
    _, reward, terminated, truncated, info = env.step(Action.DIAGNOSE_CANCEROUS)
    assert terminated and not truncated
    assert reward == pytest.approx(0.999)
    assert not info["action_mask"].any()


def test_cross_scale_evidence_is_unique_per_cell_and_scale() -> None:
    env = make_test_env(min_diagnostic_evidence=1)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    env.step(Action.PAN_UP)
    env.step(Action.PAN_LEFT)
    env.step(Action.ZOOM_IN_TL)
    _, _, _, _, info = env.step(Action.ZOOM_IN_TL)
    assert info["evidence_count"] == 1
    _, _, _, _, info = env.step(Action.ZOOM_OUT)
    _, _, _, _, info = env.step(Action.ZOOM_IN_TL)
    assert info["evidence_count"] == 1
    _, _, _, _, info = env.step(Action.ZOOM_IN_TL)
    assert info["evidence_count"] == 2


def test_reset_clears_belief_before_recording_trajectory() -> None:
    env = make_test_env()
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    env.step(Action.PAN_UP)
    env.step(Action.PAN_LEFT)
    env.step(Action.ZOOM_IN_TL)
    env.step(Action.ZOOM_IN_TL)
    assert env.unwrapped.belief_state.probability != pytest.approx(0.5)
    env.reset(seed=2, options={"slide_id": "synthetic_benign"})
    assert env.unwrapped.trajectory[0]["belief"] == pytest.approx(0.5)


def test_first_evidence_updates_belief_without_belief_shaping() -> None:
    env = make_test_env(reward_type="belief", belief_reward_beta=1.0)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})

    reward, info = advance_to_first_evidence(env)

    assert info["evidence_added"]
    assert info["evidence_count"] == 1
    assert info["belief"] > 0.5
    assert reward == pytest.approx(-0.001)


def test_belief_shaping_begins_with_second_evidence() -> None:
    env = make_test_env(reward_type="belief", belief_reward_beta=1.0)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})
    _, first_info = advance_to_first_evidence(env)

    _, reward, _, _, second_info = env.step(Action.ZOOM_IN_TL)

    assert second_info["evidence_added"]
    assert second_info["evidence_count"] == 2
    assert reward == pytest.approx(-0.001 + second_info["belief"] - first_info["belief"])


def test_default_navigation_reward_unchanged_for_first_and_second_evidence() -> None:
    env = make_test_env(reward_type="default", belief_reward_beta=10.0)
    env.reset(seed=1, options={"slide_id": "synthetic_cancer"})

    first_reward, first_info = advance_to_first_evidence(env)
    _, second_reward, _, _, second_info = env.step(Action.ZOOM_IN_TL)

    assert first_info["evidence_added"] and second_info["evidence_added"]
    assert first_reward == pytest.approx(-0.001)
    assert second_reward == pytest.approx(-0.001)
