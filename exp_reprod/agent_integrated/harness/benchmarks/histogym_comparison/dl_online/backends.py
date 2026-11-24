"""Native backend adapters used by the shared recurrent-PPO runner."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import gymnasium
import numpy as np

from benchmarks.histogym_comparison.dl_online.common import EpisodeSpec
from benchmarks.histogym_comparison.wsi.cohort import CohortSelection, load_cohort

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/camelyon16.yaml"
HISTOGYM_ROOT = ROOT / "benchmarks/histogym_comparison/histogym/upstream/HistoGym"


@dataclass(frozen=True, slots=True)
class ResetResult:
    observation: Any
    info: dict[str, Any]
    reset_ns: int
    construction_ns: int
    close_ns: int


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    info: dict[str, Any]
    step_ns: int
    state_changed: bool
    embedding_lookup_ns: int
    belief_inference_ns: int
    evidence_added: bool


class TimedEpisodeSource:
    def __init__(self, source: Any) -> None:
        self.source = source
        self.embedding_lookup_ns = 0
        self.embedding_reads = 0

    def slide_ids(self, split: str):
        return self.source.slide_ids(split)

    def load_slide(self, slide_id: str):
        return self.source.load_slide(slide_id)

    def read_viewport(self, slide_id: str, cell: Any, output_size: int):
        return self.source.read_viewport(slide_id, cell, output_size)

    def read_embedding(self, slide_id: str, cell: Any):
        started = time.perf_counter_ns()
        try:
            return self.source.read_embedding(slide_id, cell)
        finally:
            self.embedding_lookup_ns += time.perf_counter_ns() - started
            self.embedding_reads += 1

    def close(self) -> None:
        self.source.close()


class TimedBeliefAccumulator:
    def __init__(self, belief: Any) -> None:
        self.belief = belief
        self.inference_ns = 0
        self.add_calls = 0

    @property
    def probability(self) -> float:
        return float(self.belief.probability)

    @property
    def evidence_count(self) -> int:
        return int(self.belief.evidence_count)

    def reset(self):
        return self.belief.reset()

    def add(self, feature: Any, magnification: float, **kwargs: Any):
        started = time.perf_counter_ns()
        try:
            return self.belief.add(feature, magnification, **kwargs)
        finally:
            self.inference_ns += time.perf_counter_ns() - started
            self.add_calls += 1


class WSILabBackend:
    name = "wsi_lab"
    num_actions = 11
    observation_shapes: ClassVar[dict[str, tuple[int, ...]]] = {
        "global": (5, 224, 224),
        "viewport": (3, 224, 224),
        "scalars": (5,),
    }

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        workspace: str | Path,
        belief_checkpoint: str | Path,
        config_path: str | Path = DEFAULT_CONFIG,
    ) -> None:
        from benchmarks.histogym_comparison.wsi.runner import (
            WorkspaceLayout,
            _load_native_context,
            _verify_prepared_workspace,
        )
        from wsi_lab.belief.accumulator import BeliefAccumulator
        from wsi_lab.belief.trainer import load_calibrated_belief
        from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
        from wsi_lab.data.splits import DatasetSplit, SplitAssignment
        from wsi_lab.envs.wsi_env import WSIEnv
        from wsi_lab.preprocessing.artifacts import ArtifactStore
        from wsi_lab.preprocessing.pipeline import ProcessedEpisodeSource
        from wsi_lab.wsi.reader import OpenSlideReader

        constructor_started = time.perf_counter_ns()
        self.selection: CohortSelection = load_cohort(manifest_path, cohort="measured")
        self.slides_by_id = {slide.slide_id: slide for slide in self.selection.slides}
        self.config_path = Path(config_path).expanduser().resolve()
        self.config, adapter = _load_native_context(self.config_path, self.selection)
        self.layout = WorkspaceLayout.from_root(workspace)
        _verify_prepared_workspace(
            self.layout,
            self.selection,
            self.config_path,
        )
        assignments = tuple(
            SplitAssignment(
                slide_id=slide.slide_id,
                label=slide.label,
                split="train",
                patient_id=None,
            )
            for slide in self.selection.slides
        )
        split = DatasetSplit(
            dataset=adapter.name,
            seed=1,
            validation_fraction=0.0,
            strategy="fixed-dl-online-benchmark-manifest",
            assignments=assignments,
        )
        reader = OpenSlideReader(
            handle_cache_size=int(self.config.environment.slide_handle_cache_size),
            viewport_cache_size=int(self.config.environment.viewport_cache_size),
            persistent_cache_dir=None,
        )
        native_source = ProcessedEpisodeSource(
            adapter,
            HDF5EmbeddingProvider(adapter),
            reader,
            ArtifactStore(self.layout.processed_root),
            split,
        )
        self.source = TimedEpisodeSource(native_source)
        belief_load_started = time.perf_counter_ns()
        belief_model, temperature, checkpoint = load_calibrated_belief(
            belief_checkpoint,
            device="cpu",
        )
        self.belief_checkpoint_load_ns = time.perf_counter_ns() - belief_load_started
        self.belief = TimedBeliefAccumulator(
            BeliefAccumulator(belief_model, temperature=temperature)
        )
        self.belief_temperature = float(temperature)
        architecture = checkpoint.get("architecture", {})
        self.belief_architecture = dict(architecture) if isinstance(architecture, dict) else {}
        self.env = WSIEnv(
            self.source,
            self.belief,
            split="train",
            evaluation=True,
            overview_size=int(self.config.environment.overview_size),
            viewport_size=int(self.config.environment.viewport_size),
            start_position=self.config.environment.start_position,
            geometric_augmentation=False,
            photometric_augmentation=False,
            max_episode_steps=int(self.config.environment.max_episode_steps),
            min_diagnostic_evidence=int(self.config.environment.min_diagnostic_evidence),
            max_visit_count=int(self.config.environment.max_visit_count),
            reward_type=self.config.environment.reward_type,
            step_cost=float(self.config.environment.step_cost),
            belief_reward_beta=float(self.config.environment.belief_reward_beta),
            lightweight_evaluation=False,
            render_mode=None,
        )
        self.initial_construction_ns = time.perf_counter_ns() - constructor_started
        self.closed = False

    @staticmethod
    def _state(info: dict[str, Any]) -> tuple[int, float]:
        return int(info["cell_id"]), float(info["magnification"])

    def action_mask(self, info: dict[str, Any]) -> np.ndarray:
        mask = np.asarray(info["action_mask"], dtype=np.bool_).copy()
        if mask.shape != (11,):
            raise ValueError(f"WSI-Lab action mask has shape {mask.shape}")
        mask[9:11] = False
        if not mask.any():
            raise RuntimeError("WSI-Lab has no valid navigation action")
        return mask

    def reset(self, episode: EpisodeSpec, *, seed: int) -> ResetResult:
        if episode.slide_id not in self.slides_by_id:
            raise KeyError(episode.slide_id)
        started = time.perf_counter_ns()
        observation, info = self.env.reset(
            seed=int(seed) + int(episode.episode_index),
            options={"slide_id": episode.slide_id},
        )
        return ResetResult(
            observation=observation,
            info=dict(info),
            reset_ns=time.perf_counter_ns() - started,
            # The one WSI-Lab constructor is nested in backend setup, which is
            # already part of the outer online timer and reported separately.
            construction_ns=0,
            close_ns=0,
        )

    def step(self, action: int, previous_info: dict[str, Any]) -> StepResult:
        lookup_before = self.source.embedding_lookup_ns
        belief_before = self.belief.inference_ns
        state_before = self._state(previous_info)
        started = time.perf_counter_ns()
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        elapsed = time.perf_counter_ns() - started
        if truncated:
            raise RuntimeError("native WSI-Lab truncated before the external 100-step limit")
        info = dict(info)
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            info=info,
            step_ns=elapsed,
            state_changed=self._state(info) != state_before,
            embedding_lookup_ns=self.source.embedding_lookup_ns - lookup_before,
            belief_inference_ns=self.belief.inference_ns - belief_before,
            evidence_added=bool(info.get("evidence_added", False)),
        )

    def close(self) -> int:
        if self.closed:
            return 0
        started = time.perf_counter_ns()
        self.env.close()
        self.closed = True
        return time.perf_counter_ns() - started

    def metadata(self) -> dict[str, Any]:
        return {
            "belief_checkpoint_load_ns": int(self.belief_checkpoint_load_ns),
            "initial_environment_construction_ns": int(self.initial_construction_ns),
            "belief_temperature": self.belief_temperature,
            "belief_architecture": self.belief_architecture,
            "belief_add_calls": int(self.belief.add_calls),
            "embedding_reads": int(self.source.embedding_reads),
            "embedding_lookup_ns": int(self.source.embedding_lookup_ns),
            "belief_inference_ns": int(self.belief.inference_ns),
            "precomputed_embeddings_only": True,
            "uni_v2_generation_included": False,
        }


class HistoGymBackend:
    name = "histogym"
    num_actions = 7
    observation_shapes: ClassVar[dict[str, tuple[int, ...]]] = {"image": (3, 224, 224)}

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        output_dir: str | Path,
        tile_size: int = 224,
    ) -> None:
        self.selection = load_cohort(manifest_path, cohort="measured")
        self.slides_by_id = {slide.slide_id: slide for slide in self.selection.slides}
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.native_root = self.output_dir / "native_results"
        self.native_root.mkdir(parents=True, exist_ok=False)
        self.tile_size = int(tile_size)
        self._resources = contextlib.ExitStack()
        self.quiet = self._resources.enter_context(
            Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115 - ExitStack owns it
        )
        import_started = time.perf_counter_ns()
        self.module = self._load_module()
        self.import_ns = time.perf_counter_ns() - import_started
        self.env: Any | None = None
        self.active_slide_id: str | None = None
        self.constructor_count = 0
        self.close_count = 0
        self.closed = False

    @staticmethod
    def _load_module():
        # The pinned source imports legacy ``gym``. Mapping that narrow import
        # to Gymnasium keeps the environment source unchanged while allowing
        # the exact same Python/PyTorch process to own both policies.
        sys.modules.setdefault("gym", gymnasium)
        sys.modules.setdefault("gym.spaces", gymnasium.spaces)
        root_text = str(HISTOGYM_ROOT)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        name = "wsi_lab_benchmark_pinned_histogym"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, HISTOGYM_ROOT / "gymhisto.py")
        if spec is None or spec.loader is None:
            raise ImportError("cannot load pinned HistoGym module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
        return module

    @staticmethod
    def _state(env: Any) -> tuple[int, float, float]:
        position = env.agent_pos
        return int(position[0]), float(position[1]), float(position[2])

    def action_mask(self, info: dict[str, Any]) -> np.ndarray:
        del info
        return np.asarray([True, True, True, True, True, True, False], dtype=np.bool_)

    def _close_active(self) -> int:
        if self.env is None:
            return 0
        started = time.perf_counter_ns()
        with contextlib.redirect_stdout(self.quiet):
            try:
                self.env.close()
            finally:
                slide = getattr(self.env, "slide", None)
                if slide is not None:
                    slide.close()
        elapsed = time.perf_counter_ns() - started
        self.env = None
        self.active_slide_id = None
        self.close_count += 1
        return elapsed

    def reset(self, episode: EpisodeSpec, *, seed: int) -> ResetResult:
        del seed  # Upstream reset has no seed parameter or stochastic state.
        row = self.slides_by_id[episode.slide_id]
        close_ns = 0
        construction_ns = 0
        if self.env is not None and self.active_slide_id != episode.slide_id:
            close_ns = self._close_active()
        if self.env is None:
            native_path = (
                self.native_root / f"episode_{episode.episode_index:05d}_{episode.slide_id}"
            )
            started = time.perf_counter_ns()
            with contextlib.redirect_stdout(self.quiet):
                self.env = self.module.HistoEnv(
                    str(row.wsi_path),
                    str(row.xml_path),
                    self.tile_size,
                    str(native_path),
                )
            construction_ns = time.perf_counter_ns() - started
            self.active_slide_id = episode.slide_id
            self.constructor_count += 1
        reset_started = time.perf_counter_ns()
        with contextlib.redirect_stdout(self.quiet):
            raw = self.env.reset()
        reset_ns = time.perf_counter_ns() - reset_started
        observation = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
        if np.asarray(observation).shape != (3, 224, 224):
            raise ValueError(f"HistoGym observation shape drifted: {np.asarray(observation).shape}")
        return ResetResult(
            observation=observation,
            info={"position": self._state(self.env)},
            reset_ns=reset_ns,
            construction_ns=construction_ns,
            close_ns=close_ns,
        )

    def step(self, action: int, previous_info: dict[str, Any]) -> StepResult:
        if self.env is None:
            raise RuntimeError("HistoGym step called without an environment")
        state_before = tuple(previous_info["position"])
        started = time.perf_counter_ns()
        with contextlib.redirect_stdout(self.quiet):
            raw = self.env.step(int(action))
        elapsed = time.perf_counter_ns() - started
        if len(raw) == 4:
            observation, reward, done, native_info = raw
        else:
            observation, reward, terminated, truncated, native_info = raw
            done = bool(terminated or truncated)
        state_after = self._state(self.env)
        info = {"position": state_after, "native_info": native_info}
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(done),
            info=info,
            step_ns=elapsed,
            state_changed=state_after != state_before,
            embedding_lookup_ns=0,
            belief_inference_ns=0,
            evidence_added=False,
        )

    def close(self) -> int:
        if self.closed:
            return 0
        elapsed = self._close_active()
        self._resources.close()
        self.closed = True
        return elapsed

    def metadata(self) -> dict[str, Any]:
        return {
            "pinned_import_ns": int(self.import_ns),
            "constructor_count": int(self.constructor_count),
            "close_count": int(self.close_count),
            "unified_python_compatibility": "legacy gym imports mapped to gymnasium",
            "pinned_source_revision": "5b742760321d28a469d2ef9494e886a49d0aa7f7",
        }


__all__ = [
    "HISTOGYM_ROOT",
    "HistoGymBackend",
    "ResetResult",
    "StepResult",
    "WSILabBackend",
]
