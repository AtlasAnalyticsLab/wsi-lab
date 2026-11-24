"""Fixed training-set monitoring at early, middle and final PPO milestones."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_cache import RasterTransform
from .rendering import render_visit_overlay
from .telemetry import aggregate_episode_records

SCHEMA_VERSION = 1
DEFAULT_MILESTONES = (100_000, 1_000_000, 2_000_000)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def cohort_sha256(cohort: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json([dict(item) for item in cohort])).hexdigest()


def monitor_reset_seed(base_seed: int, slide_id: str) -> int:
    """Stable per-slide seed shared by configurations and milestones."""

    digest = hashlib.sha256(f"{int(base_seed)}\0{slide_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def write_trajectory_gzip(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Write deterministic gzip JSON so raw visits can regenerate every overlay."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(document)) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
                stream.write(payload)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def read_trajectory_gzip(path: str | Path) -> dict[str, Any]:
    with gzip.open(Path(path).expanduser().resolve(), "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"incompatible trajectory artifact: {path}")
    return value


def _trajectory_point(
    info: Mapping[str, Any],
    *,
    step: int,
    action: int | None,
    reward: float | None,
    is_view: bool,
    terminated: bool = False,
    truncated: bool = False,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_bounds = info.get("canonical_cell_bounds_level0", info.get("canonical_bounds"))
    if raw_bounds is None:
        raise KeyError("monitor info requires canonical cell bounds")
    bounds = [int(value) for value in raw_bounds]
    if len(bounds) != 4:
        raise ValueError("monitor bounds must contain four values")
    return {
        "step": int(step),
        "action": None if action is None else int(action),
        "is_view": bool(is_view),
        "cell_id": None if info.get("cell_id") is None else int(info["cell_id"]),
        "magnification": (
            None if info.get("magnification") is None else float(info["magnification"])
        ),
        "canonical_bounds": bounds,
        "belief": None if info.get("belief") is None else float(info["belief"]),
        "evidence_count": (
            None if info.get("evidence_count") is None else int(info["evidence_count"])
        ),
        "reward": None if reward is None else float(reward),
        "invalid_action": bool(info.get("invalid_action", False)),
        "white_viewport": (
            None if info.get("white_viewport") is None else bool(info["white_viewport"])
        ),
        "tumor_fraction": (
            None if info.get("tumor_fraction") is None else float(info["tumor_fraction"])
        ),
        "tissue_fraction": (
            None if info.get("tissue_fraction") is None else float(info["tissue_fraction"])
        ),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "policy": dict(policy or {}),
    }


def run_monitor_episode(
    env: Any,
    select_action: Callable[[Any, Mapping[str, Any], bool], int | tuple[int, Mapping[str, Any]]],
    *,
    slide_id: str,
    seed: int,
    navigation_actions: Iterable[int] = range(9),
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one forced-slide episode with a caller-owned deterministic selector.

    ``select_action`` may be a closure over a recurrent model state.  It receives
    ``(observation, info, episode_start)`` and returns an action or
    ``(action, policy_metadata)``.  The helper never touches global RNG state.
    """

    navigation = frozenset(int(value) for value in navigation_actions)
    observation, info = env.reset(seed=int(seed), options={"slide_id": str(slide_id)})
    trajectory = [
        _trajectory_point(
            info,
            step=0,
            action=None,
            reward=None,
            is_view=True,
        )
    ]
    total_return = 0.0
    episode_length = 0
    episode_start = True
    terminal_info: Mapping[str, Any] = info
    while True:
        selection = select_action(observation, info, episode_start)
        if isinstance(selection, tuple):
            action, policy = int(selection[0]), dict(selection[1])
        else:
            action, policy = int(selection), {}
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        episode_length += 1
        total_return += float(reward)
        is_view = action in navigation and not bool(next_info.get("invalid_action", False))
        trajectory.append(
            _trajectory_point(
                next_info,
                step=episode_length,
                action=action,
                reward=float(reward),
                is_view=is_view,
                terminated=bool(terminated),
                truncated=bool(truncated),
                policy=policy,
            )
        )
        if terminated or truncated:
            terminal_info = next_info
            break
        if max_steps is not None and episode_length >= int(max_steps):
            raise RuntimeError("monitor episode exceeded max_steps without a Gym terminal flag")
        observation, info = next_observation, next_info
        episode_start = False
    metrics = terminal_info.get("episode_metrics")
    details = (
        {**terminal_info, **dict(metrics)} if isinstance(metrics, Mapping) else dict(terminal_info)
    )
    diagnosis = details.get("diagnosis", details.get("prediction"))
    label = details.get("label")
    correct = details.get("correct")
    if correct is None and diagnosis is not None and label is not None:
        correct = int(diagnosis) == int(label)
    summary = {
        "slide_id": str(slide_id),
        "label": None if label is None else int(label),
        "diagnosis": None if diagnosis is None else int(diagnosis),
        "correct": None if correct is None else bool(correct),
        "episode_return": total_return,
        "episode_length": episode_length,
        "terminated": bool(trajectory[-1]["terminated"]),
        "truncated": bool(trajectory[-1]["truncated"]),
        "termination_reason": details.get("termination_reason"),
        "classifier_source": details.get("classifier_source"),
        "final_belief": details.get("final_belief", details.get("belief")),
    }
    for name in (
        "revisit_rate",
        "tissue_coverage",
        "tissue_hit_rate",
        "white_view_fraction",
        "tumor_annotation_available",
        "tumor_annotation_exhaustive",
        "tumor_hit",
        "tumor_view_fraction",
        "tumor_coverage",
        "first_tumor_step",
        "action_histogram",
        "magnification_histogram",
        "diagnosis_step",
        "early_diagnosis",
    ):
        if name in details:
            summary[name] = details[name]
    return {"trajectory": trajectory, "episode_summary": summary}


def available_diagnosis_cancer_score(
    logits: Sequence[float] | np.ndarray,
    available: Sequence[bool] | np.ndarray,
) -> float | None:
    """Cancer probability conditional on currently available diagnosis actions.

    Both diagnosis actions are normally masked until the minimum-evidence gate
    opens.  Returning ``None`` in that state avoids applying softmax to two
    negative infinities.  If only benign is available, the cancer score is 0.
    """

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    mask = np.asarray(available, dtype=np.bool_).reshape(-1)
    if values.shape != (2,) or mask.shape != (2,):
        raise ValueError("diagnosis logits and availability must each have length two")
    if not bool(mask.any()):
        return None
    selected = values[mask]
    if not bool(np.isfinite(selected).all()):
        return None
    weights = np.exp(selected - float(np.max(selected)))
    probabilities = weights / float(weights.sum())
    if not bool(mask[0]):
        return 0.0
    cancer_index = int(np.flatnonzero(mask).tolist().index(0))
    return float(probabilities[cancer_index])


class MilestoneBookkeeper:
    """Crash-safe monitoring manifest and resume reconciliation."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        milestones: Sequence[int] = DEFAULT_MILESTONES,
        variant_id: str,
        cohort_digest: str,
    ) -> None:
        values = tuple(sorted({int(value) for value in milestones}))
        if not values or values[0] <= 0:
            raise ValueError("monitor milestones must be positive")
        self.path = Path(manifest_path).expanduser().resolve()
        self.milestones = values
        self.variant_id = str(variant_id)
        self.cohort_digest = str(cohort_digest)
        self.completed: list[dict[str, Any]] = []
        if self.path.is_file():
            self._load()
        else:
            self._write("in_progress")

    def _load(self) -> None:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("variant_id") != self.variant_id
            or tuple(value.get("expected_milestones", ())) != self.milestones
            or value.get("cohort_sha256") != self.cohort_digest
        ):
            raise ValueError(f"incompatible monitoring manifest: {self.path}")
        self.completed = [dict(item) for item in value.get("completed", ())]

    def _write(self, status: str) -> None:
        _atomic_json(
            self.path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "variant_id": self.variant_id,
                "expected_milestones": list(self.milestones),
                "cohort_sha256": self.cohort_digest,
                "completed": self.completed,
            },
        )

    def reconcile(self, step: int) -> None:
        restored = int(step)
        self.completed = [
            item for item in self.completed if int(item.get("actual_step", -1)) <= restored
        ]
        self._write("in_progress")

    def due(self, step: int) -> tuple[int, ...]:
        finished = {int(item["nominal_step"]) for item in self.completed}
        return tuple(
            value for value in self.milestones if value <= int(step) and value not in finished
        )

    def mark_complete(self, record: Mapping[str, Any]) -> None:
        item = dict(record)
        nominal = int(item["nominal_step"])
        if nominal not in self.milestones:
            raise ValueError(f"unknown monitoring milestone: {nominal}")
        self.completed = [
            existing for existing in self.completed if int(existing["nominal_step"]) != nominal
        ]
        self.completed.append(item)
        self.completed.sort(key=lambda value: int(value["nominal_step"]))
        self._write("in_progress")

    def finalize(self) -> dict[str, Any]:
        finished = {int(item["nominal_step"]) for item in self.completed}
        if finished != set(self.milestones):
            raise RuntimeError(
                f"monitoring milestones incomplete: {sorted(set(self.milestones) - finished)}"
            )
        self._write("complete")
        return json.loads(self.path.read_text(encoding="utf-8"))


class TrainingSetMonitor:
    """Capture the same fixed train slides whenever a threshold is crossed."""

    def __init__(
        self,
        output_dir: str | Path,
        cohort: Sequence[Mapping[str, Any]],
        variant_id: str,
        *,
        milestones: Sequence[int] = DEFAULT_MILESTONES,
        start_strategy: str,
        reset_seed: int,
        episode_runner: Callable[[Any, Any, str, int], Mapping[str, Any]],
        overview_provider: Callable[[str], tuple[np.ndarray, RasterTransform]],
        tumor_overlay_provider: Callable[[str], np.ndarray | None] | None = None,
    ) -> None:
        if not cohort:
            raise ValueError("monitor cohort cannot be empty")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cohort = tuple(dict(item) for item in cohort)
        self.variant_id = str(variant_id)
        self.start_strategy = str(start_strategy)
        self.reset_seed = int(reset_seed)
        self.episode_runner = episode_runner
        self.overview_provider = overview_provider
        self.tumor_overlay_provider = tumor_overlay_provider
        self.bookkeeper = MilestoneBookkeeper(
            self.output_dir / "manifest.json",
            milestones=milestones,
            variant_id=self.variant_id,
            cohort_digest=cohort_sha256(self.cohort),
        )

    @property
    def manifest_path(self) -> Path:
        return self.bookkeeper.path

    def reconcile(self, step: int) -> None:
        self.bookkeeper.reconcile(step)

    def _capture_milestone(
        self,
        model: Any,
        *,
        nominal_step: int,
        actual_step: int,
        env_factory: Callable[[], Any],
    ) -> dict[str, Any]:
        root = self.output_dir / f"step_{nominal_step:012d}"
        environment = env_factory()
        summaries: list[dict[str, Any]] = []
        try:
            for cohort_record in self.cohort:
                slide_id = str(cohort_record["slide_id"])
                seed = monitor_reset_seed(self.reset_seed, slide_id)
                result = dict(self.episode_runner(model, environment, slide_id, seed))
                trajectory = list(result["trajectory"])
                summary = dict(result.get("episode_summary", {}))
                overview, transform = self.overview_provider(slide_id)
                tumor_overlay = (
                    self.tumor_overlay_provider(slide_id)
                    if self.tumor_overlay_provider is not None
                    else None
                )
                slide_root = root / slide_id
                raw_path = slide_root / "trajectory.json.gz"
                overlay_path = slide_root / "overlay.png"
                document = {
                    "schema_version": SCHEMA_VERSION,
                    "variant_id": self.variant_id,
                    "start_strategy": self.start_strategy,
                    "nominal_step": int(nominal_step),
                    "actual_step": int(actual_step),
                    "slide_id": slide_id,
                    "label": int(cohort_record["label"]),
                    "reset_seed": seed,
                    "overview_transform": {
                        name: int(getattr(transform, name))
                        for name in (
                            "source_width",
                            "source_height",
                            "canvas_width",
                            "canvas_height",
                            "content_x0",
                            "content_y0",
                            "content_width",
                            "content_height",
                        )
                    },
                    "cohort_record": cohort_record,
                    "episode_summary": summary,
                    "trajectory": trajectory,
                }
                write_trajectory_gzip(raw_path, document)
                render_visit_overlay(
                    overview,
                    transform,
                    trajectory,
                    overlay_path,
                    tumor_overlay=tumor_overlay,
                    title=f"{slide_id} — {self.variant_id} — step {actual_step:,}",
                )
                summary.update(
                    {
                        "slide_id": slide_id,
                        "label": int(cohort_record["label"]),
                        "trajectory": str(raw_path),
                        "overlay": str(overlay_path),
                    }
                )
                summaries.append(summary)
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        aggregate = aggregate_episode_records(summaries)
        summary_path = root / "summary.json"
        _atomic_json(
            summary_path,
            {
                "schema_version": SCHEMA_VERSION,
                "variant_id": self.variant_id,
                "nominal_step": int(nominal_step),
                "actual_step": int(actual_step),
                "slides": summaries,
                "aggregate": aggregate,
            },
        )
        return {
            "nominal_step": int(nominal_step),
            "actual_step": int(actual_step),
            "slides": len(summaries),
            "output_dir": str(root),
            "summary": str(summary_path),
        }

    def maybe_run(
        self,
        model: Any,
        step: int,
        env_factory: Callable[[], Any],
    ) -> tuple[dict[str, Any], ...]:
        from wsi_lab.rl.checkpoints import capture_rng_state, restore_rng_state

        completed: list[dict[str, Any]] = []
        was_training = bool(getattr(model, "training", False))
        # Monitoring is observational.  Production monitor construction loads
        # a fresh belief module and rendering/helpers may consume process RNG;
        # without restoration, milestone callbacks would perturb subsequent
        # PPO action sampling and minibatch shuffles (and do so differently for
        # belief-disabled variants).
        rng_state = capture_rng_state()
        evaluation_mode = getattr(model, "eval", None)
        if callable(evaluation_mode):
            evaluation_mode()
        try:
            for nominal in self.bookkeeper.due(step):
                record = self._capture_milestone(
                    model,
                    nominal_step=nominal,
                    actual_step=int(step),
                    env_factory=env_factory,
                )
                self.bookkeeper.mark_complete(record)
                completed.append(record)
        finally:
            restore_rng_state(rng_state)
            training_mode = getattr(model, "train", None)
            if callable(training_mode):
                training_mode(was_training)
        return tuple(completed)

    def finalize(self, step: int) -> dict[str, Any]:
        if self.bookkeeper.due(step):
            raise RuntimeError("due monitoring milestones must be captured before finalize")
        return self.bookkeeper.finalize()


class _BoundTrainingMonitor:
    """Runner-facing adapter with its production environment factory bound."""

    def __init__(self, monitor: TrainingSetMonitor, env_factory: Callable[[], Any]) -> None:
        self.monitor = monitor
        self.env_factory = env_factory

    @property
    def manifest_path(self) -> Path:
        return self.monitor.manifest_path

    def reconcile(self, step: int) -> None:
        self.monitor.reconcile(step)

    def maybe_run(self, model: Any, step: int) -> tuple[dict[str, Any], ...]:
        return self.monitor.maybe_run(model, step, self.env_factory)

    def finalize(self, step: int, model: Any) -> dict[str, Any]:
        self.maybe_run(model, step)
        return self.monitor.finalize(step)


def build_training_monitor(
    *,
    output_dir: str | Path,
    cohort: Sequence[Mapping[str, Any]],
    variant_id: str,
    mode: str,
    config: Any,
    analysis_cache: Any,
) -> _BoundTrainingMonitor:
    """Bind the study runner's production train-only deterministic monitor."""

    import torch

    from wsi_lab.cli import _preprocessing_store
    from wsi_lab.rl.rollout import observation_to, pack_observation

    from .telemetry import EpisodeTelemetryWrapper
    from .variants import (
        get_variant_spec,
        make_variant_environment,
    )

    if mode not in {"main", "smoke"}:
        raise ValueError("monitor mode must be 'main' or 'smoke'")
    spec = get_variant_spec(variant_id)
    milestones = DEFAULT_MILESTONES if mode == "main" else (1_024,)
    slide_ids = tuple(str(item["slide_id"]) for item in cohort)
    store = _preprocessing_store(config)

    def env_factory() -> EpisodeTelemetryWrapper:
        environment = make_variant_environment(
            config,
            spec,
            split_name="train",
            evaluation=True,
            allow_test=False,
            slide_ids=slide_ids,
        )
        return EpisodeTelemetryWrapper(
            environment,
            analysis_provider=analysis_cache,
            navigation_actions=range(9),
        )

    def episode_runner(model: Any, env: Any, slide_id: str, seed: int) -> Mapping[str, Any]:
        device = next(model.parameters()).device
        hidden = model.initial_state(1, device=device)

        def select_action(
            observation: Mapping[str, Any], info: Mapping[str, Any], episode_start: bool
        ) -> tuple[int, Mapping[str, Any]]:
            nonlocal hidden
            batched = {
                name: np.expand_dims(np.asarray(value), axis=0)
                for name, value in observation.items()
            }
            packed = pack_observation(batched)
            mask = np.asarray(info["action_mask"], dtype=np.bool_)
            with torch.no_grad():
                output = model.get_action_and_value(
                    observation_to(packed, device),
                    hidden,
                    torch.as_tensor([episode_start], dtype=torch.bool, device=device),
                    torch.as_tensor(mask[None, :], dtype=torch.bool, device=device),
                    deterministic=True,
                )
            hidden = output.hidden_state
            probabilities = torch.softmax(output.logits[0], dim=-1)
            action = int(output.action.item())
            policy = {
                "selected_probability": float(probabilities[action].item()),
                "value": float(output.value.item()),
            }
            if spec.diagnosis_actions:
                cancer_score = available_diagnosis_cancer_score(
                    output.logits[0, 9:11].detach().cpu().numpy(),
                    mask[9:11],
                )
                if cancer_score is not None:
                    policy["cancer_score"] = cancer_score
            return action, policy

        return run_monitor_episode(
            env,
            select_action,
            slide_id=slide_id,
            seed=seed,
            navigation_actions=range(9),
            max_steps=int(config.environment.max_episode_steps) + 1,
        )

    def overview_provider(slide_id: str) -> tuple[np.ndarray, RasterTransform]:
        metadata = store.load_metadata(slide_id)
        return (
            np.asarray(store.load_overview(slide_id, mmap=False), dtype=np.uint8),
            RasterTransform.from_mapping(metadata["overview_transform"]),
        )

    def tumor_overlay_provider(slide_id: str) -> np.ndarray | None:
        analysis = analysis_cache(slide_id)
        return None if analysis is None else analysis.tumor_overview_mask

    monitor = TrainingSetMonitor(
        Path(output_dir).expanduser().resolve() / "monitoring",
        cohort,
        variant_id,
        milestones=milestones,
        start_strategy=str(config.environment.start_position),
        reset_seed=20_260_831,
        episode_runner=episode_runner,
        overview_provider=overview_provider,
        tumor_overlay_provider=tumor_overlay_provider,
    )
    return _BoundTrainingMonitor(monitor, env_factory)


__all__ = [
    "DEFAULT_MILESTONES",
    "MilestoneBookkeeper",
    "TrainingSetMonitor",
    "available_diagnosis_cancer_score",
    "build_training_monitor",
    "cohort_sha256",
    "monitor_reset_seed",
    "read_trajectory_gzip",
    "run_monitor_episode",
    "write_trajectory_gzip",
]
