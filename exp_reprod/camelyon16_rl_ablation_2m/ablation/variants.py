#!/usr/bin/env python3
"""Study-local WSI-Lab ablation contracts.

This module deliberately leaves :mod:`wsi_lab` unchanged.  It preserves the
core trainer's fixed observation and 11-wide action-mask storage contracts,
while restricting the information branches and environment behavior selected
by each paper variant.

Public API
----------
``VARIANT_SPECS``
    Immutable-by-convention mapping of the six stable variant IDs.
``get_variant_spec(variant)``
    Resolve a variant ID or return an already-resolved :class:`VariantSpec`.
``AblationActorCritic(spec)``
    Core-compatible recurrent actor-critic that encodes only permitted inputs.
``make_variant_environment(config, spec, ...)``
    Construct either a production CAMELYON environment or a source-injected
    synthetic environment without duplicating PPO training code.
``variant_dependencies(spec)``
    Stable checkpoint dependency fields required by study resume and verification.

The core trainer and rollout buffer retain their eleven-wide compatibility
contract.  V4 and V6 own a nine-row actor head; two non-trainable constants are
padded only when constructing the masked distribution, where those entries are
permanently invalid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any, Final

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn

from wsi_lab.belief.accumulator import BeliefAccumulator
from wsi_lab.constants import (
    BELIEF_MAGNIFICATIONS,
    LEVEL0_FOOTPRINTS,
    NAVIGATION_MAGNIFICATIONS,
    NUM_ACTIONS,
)
from wsi_lab.envs.actions import Action
from wsi_lab.envs.rewards import diagnosis_reward
from wsi_lab.envs.wsi_env import (
    BeliefState,
    EpisodeSource,
    LoadedSlide,
    NavigationCell,
    WSIEnv,
)
from wsi_lab.rl.distributions import MaskedCategorical
from wsi_lab.rl.networks import (
    GLOBAL_CHANNELS,
    SCALAR_SIZE,
    VIEWPORT_CHANNELS,
    PolicyStep,
    RecurrentActorCritic,
    SequenceEvaluation,
)
from wsi_lab.utils.fingerprints import fingerprint

from .telemetry import is_white_viewport

IMPLEMENTATION_VERSION: Final = "2"
RANDOM_START_IMPLEMENTATION: Final = "vectorized_coarse_grid_v1"
COMPATIBILITY_BELIEF_VALUE: Final = 0.5
V3_WHITE_PIXEL_THRESHOLD: Final = 0.98
V3_WHITE_FRACTION_THRESHOLD: Final = 0.99


class HorizonClassifier(StrEnum):
    """Readout performed after the final navigation action."""

    NONE = "none"
    BELIEF = "belief"


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Decision-complete behavior and policy-input contract for one variant."""

    id: str
    description: str
    observe_global: bool
    observe_viewport: bool
    observe_scalars: bool
    observe_belief: bool
    tissue_action_masking: bool
    diagnosis_actions: bool
    belief_enabled: bool
    horizon_classifier: HorizonClassifier

    def __post_init__(self) -> None:
        if not self.observe_viewport:
            raise ValueError("every study variant must observe the RGB viewport")
        if self.observe_belief and not self.observe_scalars:
            raise ValueError("belief can only be observed through the scalar branch")
        if self.observe_belief and not self.belief_enabled:
            raise ValueError("a disabled belief model cannot be observed")
        if self.diagnosis_actions and self.horizon_classifier is not HorizonClassifier.NONE:
            raise ValueError("explicit diagnosis actions cannot also use a horizon classifier")
        if self.horizon_classifier is HorizonClassifier.BELIEF and not self.belief_enabled:
            raise ValueError("belief horizon classification requires the belief model")

    @property
    def behavior_num_actions(self) -> int:
        return NUM_ACTIONS if self.diagnosis_actions else len(range(9))

    @property
    def scalar_indices(self) -> tuple[int, ...]:
        """Indices supplied to the scalar encoder from the fixed five-vector."""

        if not self.observe_scalars:
            return ()
        # [magnification, remaining budget, belief, roi_x, roi_y]
        return (0, 1, 2, 3, 4) if self.observe_belief else (0, 1, 3, 4)

    @property
    def policy_branches(self) -> tuple[str, ...]:
        branches: list[str] = []
        if self.observe_global:
            branches.append("global")
        if self.observe_viewport:
            branches.append("viewport")
        if self.observe_scalars:
            branches.append("scalars")
        return tuple(branches)


VARIANT_SPECS: Final[dict[str, VariantSpec]] = {
    "v1_full": VariantSpec(
        id="v1_full",
        description="Full WSI-Lab pipeline",
        observe_global=True,
        observe_viewport=True,
        observe_scalars=True,
        observe_belief=True,
        tissue_action_masking=True,
        diagnosis_actions=True,
        belief_enabled=True,
        horizon_classifier=HorizonClassifier.NONE,
    ),
    "v2_viewport_only": VariantSpec(
        id="v2_viewport_only",
        description="Viewport-only policy observation",
        observe_global=False,
        observe_viewport=True,
        observe_scalars=False,
        observe_belief=False,
        tissue_action_masking=True,
        diagnosis_actions=True,
        belief_enabled=True,
        horizon_classifier=HorizonClassifier.NONE,
    ),
    "v3_no_tissue_mask": VariantSpec(
        id="v3_no_tissue_mask",
        description=(
            "Full observation with geometric, non-tissue-masked navigation and "
            "RGB-white evidence suppression"
        ),
        observe_global=True,
        observe_viewport=True,
        observe_scalars=True,
        observe_belief=True,
        tissue_action_masking=False,
        diagnosis_actions=True,
        belief_enabled=True,
        horizon_classifier=HorizonClassifier.NONE,
    ),
    "v4_no_diagnosis": VariantSpec(
        id="v4_no_diagnosis",
        description="Nine navigation actions with belief classification at the horizon",
        observe_global=True,
        observe_viewport=True,
        observe_scalars=True,
        observe_belief=True,
        tissue_action_masking=True,
        diagnosis_actions=False,
        belief_enabled=True,
        horizon_classifier=HorizonClassifier.BELIEF,
    ),
    "v5_no_belief": VariantSpec(
        id="v5_no_belief",
        description="Full policy inputs except belief, with no belief feature reads",
        observe_global=True,
        observe_viewport=True,
        observe_scalars=True,
        observe_belief=False,
        tissue_action_masking=True,
        diagnosis_actions=True,
        belief_enabled=False,
        horizon_classifier=HorizonClassifier.NONE,
    ),
    "v6_combined": VariantSpec(
        id="v6_combined",
        description="Viewport-only, geometric navigation, no diagnosis, and no belief",
        observe_global=False,
        observe_viewport=True,
        observe_scalars=False,
        observe_belief=False,
        tissue_action_masking=False,
        diagnosis_actions=False,
        belief_enabled=False,
        horizon_classifier=HorizonClassifier.NONE,
    ),
}


def get_variant_spec(variant: str | VariantSpec) -> VariantSpec:
    if isinstance(variant, VariantSpec):
        return variant
    try:
        return VARIANT_SPECS[str(variant)]
    except KeyError as exc:
        raise KeyError(
            f"unknown ablation variant {variant!r}; expected one of {tuple(VARIANT_SPECS)}"
        ) from exc


def variant_dependencies(variant: str | VariantSpec) -> dict[str, str]:
    """Return resume-critical dependency fields for a study checkpoint."""

    spec = get_variant_spec(variant)
    return {
        "ablation_variant_id": spec.id,
        "ablation_contract_sha256": fingerprint(asdict(spec)),
        "ablation_implementation_version": IMPLEMENTATION_VERSION,
    }


def _coarse_array(cells: Mapping[int, NavigationCell]) -> np.ndarray | None:
    """Return the mmap-backed coarse level without materializing cell objects."""

    arrays = getattr(cells, "_arrays", None)
    if not isinstance(arrays, tuple):
        return None
    for array in arrays:
        if len(array) and np.isclose(
            float(array["magnification"][0]), NAVIGATION_MAGNIFICATIONS[0]
        ):
            return array
    return None


def candidate_ids(slide: LoadedSlide, tissue_only: bool) -> tuple[int, ...]:
    """Return valid 0.625x start IDs using the optimized production path.

    ``tissue_only=False`` never reads ``tissue_valid`` or any other tissue-mask
    field.  This is the random/center start universe for V3 and V6.  Custom
    synthetic sources fall back to their mapping API; production
    ``NavigationCellMapping`` uses one vectorized read of its first mmap grid.
    """

    array = _coarse_array(slide.cells)
    if array is not None:
        selected = array
        if tissue_only:
            selected = selected[selected["tissue_valid"].astype(bool)]
        identifiers = tuple(map(int, selected["cell_id"]))
    elif tissue_only:
        identifiers = tuple(
            sorted(
                cell.cell_id
                for cell in slide.cells.values()
                if cell.magnification == NAVIGATION_MAGNIFICATIONS[0] and cell.tissue_valid
            )
        )
    else:
        reference = slide.cells[slide.start_cell_id]
        footprint = LEVEL0_FOOTPRINTS[NAVIGATION_MAGNIFICATIONS[0]]
        columns = math.ceil(slide.overview_transform.source_width / footprint)
        rows = math.ceil(slide.overview_transform.source_height / footprint)
        offset = reference.cell_id - reference.row * columns - reference.column
        identifiers = tuple(range(offset, offset + rows * columns))
    if not identifiers:
        qualifier = "tissue-valid " if tissue_only else ""
        raise ValueError(f"slide has no {qualifier}0.625x start cells")
    return identifiers


def random_start_contract(variant: str | VariantSpec) -> dict[str, str | bool]:
    """Describe the candidate domain and tissue access allowed by a variant."""

    spec = get_variant_spec(variant)
    tissue_only = spec.tissue_action_masking
    return {
        "enabled": True,
        "implementation": RANDOM_START_IMPLEMENTATION,
        "candidate_domain": ("tissue_valid_coarse" if tissue_only else "geometry_valid_coarse"),
        "uses_tissue_mask": tissue_only,
    }


def _initialize_new_module(module: nn.Module) -> None:
    """Initialize study-added projections like the core network."""

    gain = math.sqrt(2.0)
    for item in module.modules():
        if isinstance(item, nn.Linear):
            nn.init.orthogonal_(item.weight, gain=gain)
            if item.bias is not None:
                nn.init.zeros_(item.bias)


class AblationActorCritic(RecurrentActorCritic):
    """Core recurrent PPO model with variant-selected observation branches.

    Environment observations retain their original fixed shapes so the shared
    rollout buffer/checkpoint implementation can be used unchanged.  Disabled
    branches never enter this model's encoder computation.
    """

    def __init__(self, variant: str | VariantSpec, *, hidden_size: int = 256) -> None:
        self.variant = get_variant_spec(variant)
        super().__init__(hidden_size=hidden_size, num_actions=NUM_ACTIONS)
        self.behavior_num_actions = self.variant.behavior_num_actions
        self.active_num_actions = self.variant.behavior_num_actions
        self.policy_observation_branches = self.variant.policy_branches
        self.scalar_indices = self.variant.scalar_indices

        if not self.variant.diagnosis_actions:
            # Keep ``num_actions=11`` as checkpoint/buffer metadata, but own
            # only nine trainable behavior rows. Two constants are padded at
            # the distribution boundary where their mask is always false.
            self.actor = nn.Linear(self.hidden_size, self.active_num_actions)
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.zeros_(self.actor.bias)

        if not self.variant.observe_global:
            self.global_encoder = nn.Identity()
        if not self.variant.observe_scalars:
            self.scalar_encoder = nn.Identity()
        elif not self.variant.observe_belief:
            self.scalar_encoder = nn.Sequential(
                nn.Linear(len(self.scalar_indices), 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
            )
            _initialize_new_module(self.scalar_encoder)

        fusion_size = 0
        if self.variant.observe_global:
            fusion_size += 256
        if self.variant.observe_viewport:
            fusion_size += 256
        if self.variant.observe_scalars:
            fusion_size += 64
        if fusion_size != 256 + 256 + 64:
            self.fusion = nn.Sequential(nn.Linear(fusion_size, 256), nn.SiLU())
            _initialize_new_module(self.fusion)

    def encode(self, observation: Mapping[str, Tensor]) -> Tensor:
        if self.variant.policy_branches == ("global", "viewport", "scalars") and (
            self.variant.observe_belief
        ):
            return super().encode(observation)

        viewport = self._branch(observation, "viewport")
        sequence_input = viewport.ndim == 5
        if sequence_input:
            time_steps, batch_size = viewport.shape[:2]
            flat_viewport = viewport.reshape(-1, *viewport.shape[2:])
        elif viewport.ndim == 4:
            batch_size = viewport.shape[0]
            time_steps = 1
            flat_viewport = viewport
        else:
            raise ValueError("viewport must have shape [B,C,H,W] or [T,B,C,H,W]")
        if flat_viewport.shape[1] != VIEWPORT_CHANNELS:
            raise ValueError(
                f"viewport branch must have {VIEWPORT_CHANNELS} channels, "
                f"got {tuple(flat_viewport.shape[1:])}"
            )

        parameter = next(self.parameters())
        branches: list[Tensor] = []
        if self.variant.observe_global:
            global_image = self._branch(observation, "global", "overview")
            expected_ndim = 5 if sequence_input else 4
            channel_axis = 2 if sequence_input else 1
            if (
                global_image.ndim != expected_ndim
                or global_image.shape[channel_axis] != GLOBAL_CHANNELS
            ):
                raise ValueError("global branch does not match the fixed observation contract")
            flat_global = global_image.reshape(-1, *global_image.shape[-3:])
            global_float = self._normalize_image(flat_global).to(dtype=parameter.dtype)
            branches.append(self.global_encoder(global_float))

        viewport_float = self._normalize_image(flat_viewport).to(dtype=parameter.dtype)
        branches.append(self.viewport_encoder(viewport_float))

        if self.variant.observe_scalars:
            scalars = self._branch(observation, "scalars")
            expected_ndim = 3 if sequence_input else 2
            if scalars.ndim != expected_ndim or scalars.shape[-1] != SCALAR_SIZE:
                raise ValueError(f"scalars must retain fixed final dimension {SCALAR_SIZE}")
            flat_scalars = scalars.reshape(-1, SCALAR_SIZE)
            selected = flat_scalars[:, self.scalar_indices].to(dtype=parameter.dtype)
            branches.append(self.scalar_encoder(selected))

        rows = flat_viewport.shape[0]
        if any(branch.shape[0] != rows for branch in branches):
            raise ValueError("active observation branches have inconsistent batch dimensions")
        fused = self.fusion(torch.cat(branches, dim=-1))
        if sequence_input:
            return fused.reshape(time_steps, batch_size, -1)
        return fused.unsqueeze(0)

    def _distribution_logits(self, recurrent: Tensor) -> Tensor:
        active = self.actor(recurrent)
        if self.active_num_actions == NUM_ACTIONS:
            return active
        padding = active.new_zeros(*active.shape[:-1], NUM_ACTIONS - self.active_num_actions)
        return torch.cat((active, padding), dim=-1)

    def _validate_distribution_mask(self, action_mask: Tensor, logits: Tensor) -> None:
        if action_mask.shape != logits.shape:
            raise ValueError(
                f"action_mask must have shape {tuple(logits.shape)}, got {tuple(action_mask.shape)}"
            )
        if self.active_num_actions < NUM_ACTIONS and bool(
            action_mask[..., self.active_num_actions :].any()
        ):
            raise ValueError(
                "navigation-only variants cannot enable compatibility diagnosis actions"
            )

    def evaluate_actions_sequence(
        self,
        observation: Mapping[str, Tensor],
        hidden_state: Tensor,
        episode_starts: Tensor,
        action_mask: Tensor,
        actions: Tensor,
    ) -> SequenceEvaluation:
        encoded = self.encode(observation)
        recurrent, final_hidden = self.recurrent_features(encoded, hidden_state, episode_starts)
        logits = self._distribution_logits(recurrent)
        self._validate_distribution_mask(action_mask, logits)
        if actions.shape != logits.shape[:-1]:
            raise ValueError(
                f"actions must have shape {tuple(logits.shape[:-1])}, got {tuple(actions.shape)}"
            )
        distribution = MaskedCategorical(logits, action_mask)
        values = self.critic(recurrent).squeeze(-1)
        return SequenceEvaluation(
            log_prob=distribution.log_prob(actions.long()),
            entropy=distribution.entropy(),
            value=values,
            logits=distribution.logits,
            hidden_state=final_hidden,
        )

    def get_action_and_value(
        self,
        observation: Mapping[str, Tensor],
        hidden_state: Tensor,
        episode_starts: Tensor,
        action_mask: Tensor,
        *,
        action: Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicyStep:
        if episode_starts.ndim != 1:
            raise ValueError("step episode_starts must have shape [B]")
        if action_mask.ndim != 2:
            raise ValueError("step action_mask must have shape [B,A]")
        encoded = self.encode(observation)
        recurrent, final_hidden = self.recurrent_features(
            encoded,
            hidden_state,
            episode_starts.unsqueeze(0),
        )
        recurrent_step = recurrent.squeeze(0)
        logits = self._distribution_logits(recurrent_step)
        self._validate_distribution_mask(action_mask, logits)
        distribution = MaskedCategorical(logits, action_mask)
        selected_action = (
            distribution.mode
            if action is None and deterministic
            else distribution.sample()
            if action is None
            else action.long()
        )
        value = self.critic(recurrent_step).squeeze(-1)
        return PolicyStep(
            action=selected_action,
            log_prob=distribution.log_prob(selected_action),
            entropy=distribution.entropy(),
            value=value,
            logits=distribution.logits,
            hidden_state=final_hidden,
        )


class _VariantEpisodeSource:
    """Delegate episode I/O while altering only agent-facing slide scope."""

    def __init__(
        self,
        source: EpisodeSource,
        variant: VariantSpec,
        *,
        slide_ids: Sequence[str] | None = None,
    ) -> None:
        self.source = source
        self.variant = variant
        self._slide_ids = None if slide_ids is None else tuple(map(str, slide_ids))

    def slide_ids(self, split: str) -> Sequence[str]:
        available = tuple(map(str, self.source.slide_ids(split)))
        if self._slide_ids is None:
            return available
        unknown = set(self._slide_ids).difference(available)
        if unknown:
            raise ValueError(
                f"requested analysis slides are outside split {split!r}: {sorted(unknown)}"
            )
        return self._slide_ids

    def load_slide(self, slide_id: str) -> LoadedSlide:
        slide = self.source.load_slide(slide_id)
        if self.variant.tissue_action_masking:
            return slide
        transform = slide.overview_transform
        return replace(
            slide,
            navigation_roi_level0=(
                0,
                0,
                int(transform.source_width),
                int(transform.source_height),
            ),
        )

    def read_viewport(self, slide_id: str, cell: NavigationCell, output_size: int) -> np.ndarray:
        return self.source.read_viewport(slide_id, cell, output_size)

    def read_embedding(self, slide_id: str, cell: NavigationCell) -> np.ndarray:
        return self.source.read_embedding(slide_id, cell)

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if callable(close):
            close()


class AblationWSIEnv(WSIEnv):
    """WSI environment implementing the six study-local ablations."""

    def __init__(
        self,
        source: EpisodeSource,
        belief: BeliefState | None,
        *,
        variant: str | VariantSpec,
        **kwargs: Any,
    ) -> None:
        self.variant = get_variant_spec(variant)
        configured_reward = kwargs.get("reward_type", "default")
        reward_type = str(getattr(configured_reward, "value", configured_reward))
        if not self.variant.belief_enabled and reward_type == "belief":
            raise ValueError("belief-shaped reward is invalid when the belief model is disabled")
        super().__init__(source, belief if self.variant.belief_enabled else None, **kwargs)
        # Core rollout storage remains 11-wide; this records the actual behavior set.
        self.behavior_action_space = gym.spaces.Discrete(self.variant.behavior_num_actions)
        self._coarse_start_cache: dict[tuple[str, bool], tuple[int, ...]] = {}

    def _coarse_start_ids(self, *, tissue_only: bool) -> tuple[int, ...]:
        assert self._slide is not None
        key = (self._slide.slide_id, bool(tissue_only))
        cached = self._coarse_start_cache.get(key)
        if cached is not None:
            return cached

        identifiers = candidate_ids(self._slide, tissue_only)
        self._coarse_start_cache[key] = identifiers
        return identifiers

    def _select_start_cell(self) -> NavigationCell:
        assert self._slide is not None
        tissue_only = self.variant.tissue_action_masking
        if self.start_position == "center" and tissue_only:
            return self._slide.cells[self._slide.start_cell_id]

        candidates = self._coarse_start_ids(tissue_only=tissue_only)
        if self.start_position == "random":
            rng = self._np_random
            assert rng is not None
            # Sampling an integer index avoids rebuilding a NumPy array from
            # the cached tuple on every short training episode.
            selected_id = candidates[int(rng.integers(0, len(candidates)))]
        else:
            footprint = LEVEL0_FOOTPRINTS[NAVIGATION_MAGNIFICATIONS[0]]
            columns = math.ceil(self._slide.overview_transform.source_width / footprint)
            center_row = min(
                math.ceil(self._slide.overview_transform.source_height / footprint) - 1,
                int((self._slide.overview_transform.source_height / 2) // footprint),
            )
            center_column = min(
                columns - 1,
                int((self._slide.overview_transform.source_width / 2) // footprint),
            )
            first = self._slide.cells[candidates[0]]
            offset = first.cell_id - first.row * columns - first.column
            selected_id = offset + center_row * columns + center_column
        selected = self._slide.cells[selected_id]
        # WSIEnv's reset assertion historically requires tissue_valid.  In the
        # no-mask variants the start is intentionally allowed to be background.
        # Always replace the compatibility copy so this branch never reads the
        # precomputed tissue flag.
        return selected if tissue_only else replace(selected, tissue_valid=True)

    def _geometric_pan_target(self, action: Action) -> int:
        assert self._slide is not None and self._cell is not None
        footprint = LEVEL0_FOOTPRINTS[self._cell.magnification]
        columns = math.ceil(self._slide.overview_transform.source_width / footprint)
        rows = math.ceil(self._slide.overview_transform.source_height / footprint)
        row, column = self._cell.row, self._cell.column
        if action is Action.PAN_UP:
            row -= 1
        elif action is Action.PAN_DOWN:
            row += 1
        elif action is Action.PAN_LEFT:
            column -= 1
        elif action is Action.PAN_RIGHT:
            column += 1
        else:
            raise ValueError(f"not a pan action: {action}")
        if not (0 <= row < rows and 0 <= column < columns):
            return -1
        offset = self._cell.cell_id - self._cell.row * columns - self._cell.column
        return offset + row * columns + column

    def _canonical_action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        if self._terminated or self.remaining_actions <= 0:
            return mask
        assert self._cell is not None

        if self.variant.diagnosis_actions and self.remaining_actions <= 1:
            mask[Action.DIAGNOSE_CANCEROUS] = True
            mask[Action.DIAGNOSE_BENIGN] = True
            return mask

        if self.variant.tissue_action_masking:
            for index, cell_id in enumerate(self._cell.neighbor_ids):
                mask[index] = self._is_tissue_valid_target(cell_id)
            for index, cell_id in enumerate(self._cell.child_ids, start=4):
                mask[index] = self._is_tissue_valid_target(cell_id)
        else:
            for action in (
                Action.PAN_UP,
                Action.PAN_DOWN,
                Action.PAN_LEFT,
                Action.PAN_RIGHT,
            ):
                mask[action] = self._geometric_pan_target(action) >= 0
            for index, cell_id in enumerate(self._cell.child_ids, start=4):
                mask[index] = cell_id >= 0
        mask[Action.ZOOM_OUT] = self._cell.parent_id >= 0

        if self.variant.diagnosis_actions and self.evidence_count >= self.min_diagnostic_evidence:
            mask[Action.DIAGNOSE_CANCEROUS] = True
            mask[Action.DIAGNOSE_BENIGN] = True
        return mask

    def _resolve_navigation(self, action: Action) -> NavigationCell:
        if self.variant.tissue_action_masking or action.value > Action.PAN_RIGHT:
            return super()._resolve_navigation(action)
        assert self._slide is not None
        target = self._geometric_pan_target(action)
        if target < 0:
            raise ValueError(f"geometric navigation action has no target: {action}")
        return self._slide.cells[target]

    def _maybe_add_evidence(self) -> bool:
        assert self._slide is not None and self._cell is not None
        if self.variant.id == "v6_combined":
            return False
        if self.variant.id == "v3_no_tissue_mask":
            white = is_white_viewport(
                {"viewport": self._viewport},
                pixel_threshold=V3_WHITE_PIXEL_THRESHOLD,
                fraction_threshold=V3_WHITE_FRACTION_THRESHOLD,
            )
            if white is not False:
                return False
        key = (self._cell.magnification, self._cell.cell_id)
        eligible = (
            self._cell.magnification in BELIEF_MAGNIFICATIONS
            and self._cell.embedding_index >= 0
            and key not in self._evidence
        )
        if not eligible:
            return False
        if self.variant.belief_enabled:
            feature = self.source.read_embedding(self._slide.slide_id, self._cell)
            self.belief_state.add(feature, self._cell.magnification)
        # With no belief this is an eligibility count only. It intentionally
        # avoids feature reads while preserving the diagnosis availability gate.
        self._evidence.add(key)
        return True

    def _observation(self) -> dict[str, np.ndarray]:
        observation = super()._observation()
        if not self.variant.belief_enabled and "scalars" in observation:
            scalars = np.asarray(observation["scalars"], dtype=np.float32).copy()
            scalars[2] = COMPATIBILITY_BELIEF_VALUE
            observation["scalars"] = scalars
        return observation

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info.update(
            {
                "ablation_variant_id": self.variant.id,
                "belief_enabled": self.variant.belief_enabled,
                "tissue_action_masking": self.variant.tissue_action_masking,
                "diagnosis_actions_enabled": self.variant.diagnosis_actions,
                "behavior_num_actions": self.variant.behavior_num_actions,
                "policy_observation_branches": self.variant.policy_branches,
            }
        )
        if not self.variant.belief_enabled:
            info["belief"] = None
        return info

    def _trajectory_point(self, action: int | None) -> dict[str, Any]:
        point = super()._trajectory_point(action)
        if not self.variant.belief_enabled:
            point["belief"] = None
        return point

    def step(self, action: int):
        observation, reward, terminated, truncated, info = super().step(action)
        if self.variant.diagnosis_actions or terminated or truncated:
            return observation, reward, terminated, truncated, info
        if self._actions_taken < self.max_episode_steps:
            return observation, reward, False, False, info

        assert self._slide is not None
        self._terminated = True
        info = self._info()
        info.update(
            {
                "invalid_action": False,
                "termination_reason": (
                    "horizon_belief_classifier"
                    if self.variant.horizon_classifier is HorizonClassifier.BELIEF
                    else "horizon_no_classifier"
                ),
                "classifier_source": self.variant.horizon_classifier.value,
            }
        )
        if self.variant.horizon_classifier is HorizonClassifier.BELIEF:
            prediction = int(float(self.belief_state.probability) >= 0.5)
            info.update(
                {
                    "diagnosis": prediction,
                    "correct": prediction == self._slide.label,
                }
            )
            reward = diagnosis_reward(prediction, self._slide.label, self.step_cost)
        else:
            info.update({"diagnosis": None, "correct": None})
        return observation, reward, True, False, info


_AUTO_BELIEF = object()


def make_variant_environment(
    config: Any,
    variant: str | VariantSpec,
    *,
    split_name: str = "train",
    evaluation: bool = False,
    allow_test: bool = False,
    slide_ids: Sequence[str] | None = None,
    source: EpisodeSource | None = None,
    belief: BeliefState | object | None = _AUTO_BELIEF,
    render_mode: str | None = None,
) -> AblationWSIEnv:
    """Build a production or source-injected study environment.

    Production calls omit ``source`` and ``belief``.  Synthetic tests may inject
    both.  For V5/V6 the belief argument is ignored and no calibrated checkpoint
    or HDF5 feature row is loaded.
    """

    spec = get_variant_spec(variant)
    if source is None:
        from wsi_lab.cli import (
            _ensure_test_access,
            _load_active_belief,
            _load_verified_split,
            _preprocessing_store,
        )
        from wsi_lab.preprocessing.pipeline import ProcessedEpisodeSource

        _ensure_test_access(split_name, allow_test)
        split = _load_verified_split(config)
        source = ProcessedEpisodeSource.from_config(
            config,
            _preprocessing_store(config),
            split,
            allow_test=allow_test,
        )
        if spec.belief_enabled:
            model, temperature, _ = _load_active_belief(
                config,
                device=config.rl.belief_device,
            )
            resolved_belief: BeliefState | None = BeliefAccumulator(model, temperature=temperature)
        else:
            resolved_belief = None
    else:
        resolved_belief = None if belief is _AUTO_BELIEF or not spec.belief_enabled else belief

    wrapped_source = _VariantEpisodeSource(source, spec, slide_ids=slide_ids)
    environment = config.environment
    return AblationWSIEnv(
        wrapped_source,
        resolved_belief,
        variant=spec,
        split=split_name,
        evaluation=evaluation,
        overview_size=int(environment.overview_size),
        viewport_size=int(environment.viewport_size),
        start_position=environment.start_position,
        geometric_augmentation=(
            bool(environment.geometric_augmentation) if not evaluation else False
        ),
        photometric_augmentation=(
            bool(environment.photometric_augmentation) if not evaluation else False
        ),
        max_episode_steps=int(environment.max_episode_steps),
        min_diagnostic_evidence=int(environment.min_diagnostic_evidence),
        max_visit_count=int(environment.max_visit_count),
        reward_type=environment.reward_type,
        step_cost=float(environment.step_cost),
        belief_reward_beta=float(environment.belief_reward_beta),
        render_mode=render_mode,
    )


__all__ = [
    "IMPLEMENTATION_VERSION",
    "RANDOM_START_IMPLEMENTATION",
    "VARIANT_SPECS",
    "AblationActorCritic",
    "AblationWSIEnv",
    "HorizonClassifier",
    "VariantSpec",
    "candidate_ids",
    "get_variant_spec",
    "make_variant_environment",
    "random_start_contract",
    "variant_dependencies",
]
