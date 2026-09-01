from __future__ import annotations

from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from wsi_lab.envs.actions import Action
from wsi_lab.envs.wsi_env import LoadedSlide, NavigationCell
from wsi_lab.testing.synthetic import DeterministicBelief, SyntheticEpisodeSource
from wsi_lab.wsi.reader import OverviewTransform


@pytest.fixture(scope="module")
def variants() -> ModuleType:
    from ablation import variants as module

    return module


def _config(
    *,
    start_position: str = "center",
    max_episode_steps: int = 20,
    min_diagnostic_evidence: int = 1,
) -> SimpleNamespace:
    environment = SimpleNamespace(
        overview_size=32,
        viewport_size=24,
        start_position=start_position,
        geometric_augmentation=False,
        photometric_augmentation=False,
        max_episode_steps=max_episode_steps,
        min_diagnostic_evidence=min_diagnostic_evidence,
        max_visit_count=10,
        reward_type="default",
        step_cost=0.001,
        belief_reward_beta=1.0,
    )
    return SimpleNamespace(environment=environment)


class CountingSource(SyntheticEpisodeSource):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.embedding_reads = 0

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
        self.embedding_reads += 1
        return super().read_embedding(slide_id, cell)


def _first_evidence_cell(source: SyntheticEpisodeSource) -> NavigationCell:
    slide = source.load_slide("synthetic_cancer")
    cell = slide.cells[slide.start_cell_id]
    for action in (
        Action.PAN_UP,
        Action.PAN_LEFT,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
    ):
        if action <= Action.PAN_RIGHT:
            cell = slide.cells[cell.neighbor_ids[action]]
        else:
            cell = slide.cells[cell.child_ids[action - Action.ZOOM_IN_TL]]
    assert cell.magnification == 2.5 and cell.embedding_index >= 0
    return cell


def _advance_to_first_evidence(environment) -> dict[str, object]:
    info: dict[str, object] = {}
    for action in (
        Action.PAN_UP,
        Action.PAN_LEFT,
        Action.ZOOM_IN_TL,
        Action.ZOOM_IN_TL,
    ):
        _, _, terminated, truncated, info = environment.step(action)
        assert not terminated and not truncated
    return info


def test_six_stable_specs_and_resume_dependencies(variants: ModuleType) -> None:
    assert tuple(variants.VARIANT_SPECS) == (
        "v1_full",
        "v2_viewport_only",
        "v3_no_tissue_mask",
        "v4_no_diagnosis",
        "v5_no_belief",
        "v6_combined",
    )
    assert variants.get_variant_spec("v4_no_diagnosis").behavior_num_actions == 9
    assert variants.get_variant_spec("v6_combined").behavior_num_actions == 9
    assert all(
        variants.get_variant_spec(name).behavior_num_actions == 11
        for name in ("v1_full", "v2_viewport_only", "v3_no_tissue_mask", "v5_no_belief")
    )
    for name in variants.VARIANT_SPECS:
        dependencies = variants.variant_dependencies(name)
        assert dependencies["ablation_variant_id"] == name
        assert dependencies["ablation_implementation_version"] == "2"
        assert len(dependencies["ablation_contract_sha256"]) == 64


@pytest.mark.parametrize(
    "variant_id",
    [
        "v1_full",
        "v2_viewport_only",
        "v3_no_tissue_mask",
        "v4_no_diagnosis",
        "v5_no_belief",
        "v6_combined",
    ],
)
def test_environment_keeps_fixed_observation_and_mask_shapes(
    variants: ModuleType, variant_id: str
) -> None:
    environment = variants.make_variant_environment(
        _config(),
        variant_id,
        source=SyntheticEpisodeSource(overview_size=32),
        belief=DeterministicBelief(),
    )
    try:
        observation, info = environment.reset(seed=3, options={"slide_id": "synthetic_cancer"})
        assert observation["global"].shape == (5, 32, 32)
        assert observation["viewport"].shape == (3, 24, 24)
        assert observation["scalars"].shape == (5,)
        assert info["action_mask"].shape == (11,)
        assert environment.action_space.n == 11
        assert (
            environment.behavior_action_space.n
            == variants.get_variant_spec(variant_id).behavior_num_actions
        )
        if variant_id in {"v5_no_belief", "v6_combined"}:
            assert observation["scalars"][2] == pytest.approx(0.5)
            assert info["belief"] is None
    finally:
        environment.close()


def test_policy_uses_only_selected_branches(variants: ModuleType) -> None:
    torch.manual_seed(4)
    viewport = torch.rand(2, 3, 32, 32)
    viewport_only = variants.AblationActorCritic("v2_viewport_only", hidden_size=16)
    combined = variants.AblationActorCritic("v6_combined", hidden_size=16)
    assert viewport_only.encode({"viewport": viewport}).shape == (1, 2, 256)
    assert combined.encode({"viewport": viewport}).shape == (1, 2, 256)

    no_belief = variants.AblationActorCritic("v5_no_belief", hidden_size=16)
    observation = {
        "global": torch.rand(2, 5, 32, 32),
        "viewport": viewport,
        "scalars": torch.rand(2, 5),
    }
    changed_belief = {key: value.clone() for key, value in observation.items()}
    changed_belief["scalars"][:, 2] = 1000.0
    torch.testing.assert_close(
        no_belief.encode(observation),
        no_belief.encode(changed_belief),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("variant_id", ["v4_no_diagnosis", "v6_combined"])
def test_navigation_only_actor_has_nine_trainable_rows_and_eleven_wide_distribution(
    variants: ModuleType, variant_id: str
) -> None:
    model = variants.AblationActorCritic(variant_id, hidden_size=16)
    assert model.actor.out_features == 9
    assert model.num_actions == 11
    raw = model._distribution_logits(torch.zeros(2, 16))
    assert raw.shape == (2, 11)
    torch.testing.assert_close(raw[:, 9:], torch.zeros(2, 2))

    observation = {
        "global": torch.rand(2, 5, 32, 32),
        "viewport": torch.rand(2, 3, 32, 32),
        "scalars": torch.rand(2, 5),
    }
    action_mask = torch.zeros(2, 11, dtype=torch.bool)
    action_mask[:, :9] = True
    step = model.get_action_and_value(
        observation,
        model.initial_state(2),
        torch.ones(2, dtype=torch.bool),
        action_mask,
    )
    assert step.logits.shape == (2, 11)
    assert bool((step.action < 9).all())

    sequence_observation = {key: value.unsqueeze(0) for key, value in observation.items()}
    evaluation = model.evaluate_actions_sequence(
        sequence_observation,
        model.initial_state(2),
        torch.ones(1, 2, dtype=torch.bool),
        action_mask.unsqueeze(0),
        step.action.unsqueeze(0),
    )
    assert evaluation.logits.shape == (1, 2, 11)

    invalid_mask = action_mask.clone()
    invalid_mask[:, 9] = True
    with pytest.raises(ValueError, match="cannot enable compatibility diagnosis"):
        model.get_action_and_value(
            observation,
            model.initial_state(2),
            torch.ones(2, dtype=torch.bool),
            invalid_mask,
        )


@pytest.mark.parametrize(
    ("variant_id", "evidence_added", "embedding_reads"),
    [
        ("v1_full", True, 1),
        ("v2_viewport_only", True, 1),
        ("v3_no_tissue_mask", True, 1),
        ("v4_no_diagnosis", True, 1),
        ("v5_no_belief", True, 0),
        ("v6_combined", False, 0),
    ],
)
def test_evidence_contracts_do_not_add_extra_gates_to_core_variants(
    variants: ModuleType,
    variant_id: str,
    evidence_added: bool,
    embedding_reads: int,
) -> None:
    source = CountingSource(overview_size=32)
    background_evidence = replace(_first_evidence_cell(source), tissue_valid=False)
    environment = variants.make_variant_environment(
        _config(),
        variant_id,
        source=source,
        belief=DeterministicBelief(),
    )
    try:
        environment.reset(seed=3, options={"slide_id": "synthetic_cancer"})
        environment._cell = background_evidence
        if variant_id == "v3_no_tissue_mask":
            environment._viewport = np.full((24, 24, 3), 128, dtype=np.uint8)
        assert environment._maybe_add_evidence() is evidence_added
        assert environment.evidence_count == int(evidence_added)
        assert source.embedding_reads == embedding_reads
    finally:
        environment.close()


def test_v5_counts_eligible_evidence_without_reading_features(variants: ModuleType) -> None:
    source = CountingSource(overview_size=32)
    environment = variants.make_variant_environment(
        _config(min_diagnostic_evidence=1),
        "v5_no_belief",
        source=source,
        belief=DeterministicBelief(),
    )
    try:
        environment.reset(seed=3, options={"slide_id": "synthetic_cancer"})
        info = _advance_to_first_evidence(environment)
        assert info["evidence_count"] == 1
        assert source.embedding_reads == 0
        assert bool(np.asarray(info["action_mask"])[9:].all())
    finally:
        environment.close()


def test_v3_uses_rgb_white_gate_without_reading_tissue_mask(
    variants: ModuleType,
) -> None:
    source = CountingSource(overview_size=32)
    environment = variants.make_variant_environment(
        _config(),
        "v3_no_tissue_mask",
        source=source,
        belief=DeterministicBelief(),
    )
    try:
        environment.reset(seed=3, options={"slide_id": "synthetic_cancer"})
        environment._cell = replace(_first_evidence_cell(source), tissue_valid=False)
        probability = environment.belief_state.probability
        environment._viewport = np.full((24, 24, 3), 255, dtype=np.uint8)
        assert not environment._maybe_add_evidence()
        assert environment.evidence_count == 0
        assert environment.belief_state.probability == probability
        assert source.embedding_reads == 0

        environment._viewport = np.full((24, 24, 3), 128, dtype=np.uint8)
        assert environment._maybe_add_evidence()
        assert environment.evidence_count == 1
        assert source.embedding_reads == 1
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("variant_id", "reason", "classifier", "diagnosis", "correct", "reward"),
    [
        ("v4_no_diagnosis", "horizon_belief_classifier", "belief", 1, True, 0.999),
        ("v6_combined", "horizon_no_classifier", "none", None, None, -0.001),
    ],
)
def test_navigation_only_horizon_semantics(
    variants: ModuleType,
    variant_id: str,
    reason: str,
    classifier: str,
    diagnosis: int | None,
    correct: bool | None,
    reward: float,
) -> None:
    environment = variants.make_variant_environment(
        _config(max_episode_steps=3, min_diagnostic_evidence=0),
        variant_id,
        source=CountingSource(overview_size=32),
        belief=DeterministicBelief(),
    )
    try:
        _, info = environment.reset(seed=2, options={"slide_id": "synthetic_cancer"})
        for step_index in range(3):
            action_mask = np.asarray(info["action_mask"])
            assert not bool(action_mask[9:].any())
            valid = np.flatnonzero(action_mask[:9])
            assert len(valid)
            _, observed_reward, terminated, truncated, info = environment.step(int(valid[0]))
            assert not truncated
            assert terminated is (step_index == 2)
        assert observed_reward == pytest.approx(reward)
        assert info["actions_taken"] == 3
        assert info["termination_reason"] == reason
        assert info["classifier_source"] == classifier
        assert info["diagnosis"] == diagnosis
        assert info["correct"] == correct
    finally:
        environment.close()


class TissueJumpSource:
    """A three-column coarse grid whose stored edge skips a background cell."""

    def __init__(self) -> None:
        overview = np.zeros((32, 32, 3), dtype=np.uint8)
        transform = OverviewTransform.fit((49152, 16384), (32, 32))
        cells = {
            0: NavigationCell(
                0,
                0.625,
                0,
                0,
                (0, 0, 16384, 16384),
                neighbor_ids=(-1, -1, -1, 2),
                tissue_valid=True,
            ),
            1: NavigationCell(
                1,
                0.625,
                0,
                1,
                (16384, 0, 32768, 16384),
                tissue_valid=False,
            ),
            2: NavigationCell(
                2,
                0.625,
                0,
                2,
                (32768, 0, 49152, 16384),
                neighbor_ids=(-1, -1, 0, -1),
                tissue_valid=True,
            ),
        }
        self.slide = LoadedSlide(
            "tissue_jump",
            1,
            overview,
            transform,
            cells,
            0,
            navigation_roi_level0=(0, 0, 16384, 16384),
        )

    def slide_ids(self, split: str) -> tuple[str]:
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


def test_no_tissue_variant_uses_full_geometry_for_roi_center_start_and_pans(
    variants: ModuleType,
) -> None:
    environment = variants.make_variant_environment(
        _config(), "v3_no_tissue_mask", source=TissueJumpSource(), belief=DeterministicBelief()
    )
    try:
        _, info = environment.reset(seed=5, options={"slide_id": "tissue_jump"})
        assert info["cell_id"] == 1
        assert info["canonical_navigation_roi_level0"] == (0, 0, 49152, 16384)
        assert bool(info["action_mask"][Action.PAN_LEFT])
        assert bool(info["action_mask"][Action.PAN_RIGHT])
        _, _, _, _, info = environment.step(Action.PAN_RIGHT)
        assert info["cell_id"] == 2
    finally:
        environment.close()


def test_no_tissue_random_start_is_seeded_over_the_full_coarse_grid(
    variants: ModuleType,
) -> None:
    environment = variants.make_variant_environment(
        _config(start_position="random"),
        "v3_no_tissue_mask",
        source=TissueJumpSource(),
        belief=DeterministicBelief(),
    )
    try:
        first = environment.reset(seed=11, options={"slide_id": "tissue_jump"})[1]
        repeated = environment.reset(seed=11, options={"slide_id": "tissue_jump"})[1]
        sampled = {
            environment.reset(seed=seed, options={"slide_id": "tissue_jump"})[1]["cell_id"]
            for seed in range(30)
        }
        assert first["cell_id"] == repeated["cell_id"]
        assert sampled == {0, 1, 2}
    finally:
        environment.close()


class ProbeArray:
    def __init__(
        self,
        magnifications: np.ndarray,
        cell_ids: np.ndarray,
        tissue: np.ndarray,
        reads: list[int],
    ) -> None:
        self.magnifications = magnifications
        self.cell_ids = cell_ids
        self.tissue = tissue
        self.reads = reads

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "magnification":
                return self.magnifications
            if key == "cell_id":
                return self.cell_ids
            if key == "tissue_valid":
                self.reads[0] += 1
                return self.tissue
            raise KeyError(key)
        selected = np.asarray(key)
        return ProbeArray(
            self.magnifications[selected],
            self.cell_ids[selected],
            self.tissue[selected],
            self.reads,
        )


class ProbeCells(dict[int, NavigationCell]):
    def __init__(self, array: ProbeArray) -> None:
        super().__init__()
        self._arrays = (array,)


def test_vectorized_candidate_helper_has_explicit_no_tissue_access(
    variants: ModuleType,
) -> None:
    tissue_reads = [0]
    array = ProbeArray(
        np.full(4, 0.625),
        np.asarray([10, 11, 12, 13]),
        np.asarray([True, False, True, False]),
        tissue_reads,
    )
    slide = LoadedSlide(
        "probe",
        0,
        np.zeros((2, 2, 3), dtype=np.uint8),
        OverviewTransform.fit((32768, 32768), (2, 2)),
        ProbeCells(array),
        10,
    )
    assert variants.candidate_ids(slide, tissue_only=False) == (10, 11, 12, 13)
    assert tissue_reads == [0]
    assert variants.candidate_ids(slide, tissue_only=True) == (10, 12)
    assert tissue_reads == [1]
    assert variants.random_start_contract("v3_no_tissue_mask") == {
        "enabled": True,
        "implementation": "vectorized_coarse_grid_v1",
        "candidate_domain": "geometry_valid_coarse",
        "uses_tissue_mask": False,
    }
    assert variants.random_start_contract("v1_full")["candidate_domain"] == ("tissue_valid_coarse")
