"""Training-set telemetry hooks that leave PPO and environment behavior unchanged."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .analysis_cache import SlideAnalysis

SCHEMA_VERSION = 3
DEFAULT_ACTION_NAMES = (
    "pan_up",
    "pan_down",
    "pan_left",
    "pan_right",
    "zoom_in_tl",
    "zoom_in_tr",
    "zoom_in_bl",
    "zoom_in_br",
    "zoom_out",
    "diagnose_cancerous",
    "diagnose_benign",
)


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return str(value)


def _gymnasium_safe_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Omit unavailable mapping leaves before Gymnasium vectorizes ``info``.

    ``SyncVectorEnv`` allocates a numeric array from the first environment that
    reports a key. Assigning ``None`` for that key in another environment then
    raises a dtype error. Missing keys are the supported representation for
    optional info: Gymnasium supplies a corresponding boolean presence mask.
    """

    result: dict[str, Any] = {}
    for key, value in info.items():
        if value is None:
            continue
        result[str(key)] = _gymnasium_safe_info(value) if isinstance(value, Mapping) else value
    return result


def _without_gymnasium_presence_masks(info: Mapping[str, Any]) -> dict[str, Any]:
    """Recover a single environment's optional values from vector ``info``."""

    result: dict[str, Any] = {}
    for key, value in info.items():
        name = str(key)
        if name.startswith("_"):
            continue
        presence = info.get(f"_{name}")
        if presence is not None and not bool(presence):
            continue
        result[name] = (
            _without_gymnasium_presence_masks(value) if isinstance(value, Mapping) else value
        )
    return result


def _append_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n" for record in records
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            result.append(value)
    return result


def _rewrite_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n" for record in records
    ).encode()
    _atomic_bytes(path, payload)


def _named_count(values: Mapping[Any, Any] | None) -> dict[str, int]:
    """Normalize a histogram while dropping Gymnasium vector-info masks.

    Gymnasium represents a nested info value such as ``pan_left: 3`` alongside
    a boolean presence flag named ``_pan_left``.  Those companion fields are
    transport metadata, not histogram categories.
    """

    if not isinstance(values, Mapping):
        return {}
    return {str(key): int(count) for key, count in values.items() if not str(key).startswith("_")}


def _validate_episode_histogram_totals(record: Mapping[str, Any]) -> None:
    checks = (
        ("action_histogram", "episode_length"),
        ("magnification_histogram", "observed_views"),
    )
    for histogram_name, expected_name in checks:
        expected = record.get(expected_name)
        if expected is None:
            continue
        histogram = _named_count(record.get(histogram_name))
        actual = sum(histogram.values())
        if actual != int(expected):
            raise ValueError(
                f"{histogram_name} total {actual} does not match {expected_name} {int(expected)}"
            )


def is_white_viewport(
    observation: Mapping[str, Any] | Any,
    *,
    pixel_threshold: float = 0.98,
    fraction_threshold: float = 0.99,
) -> bool | None:
    """White/background test for normalized or uint8 RGB viewports."""

    if not isinstance(observation, Mapping) or "viewport" not in observation:
        return None
    viewport = np.asarray(observation["viewport"])
    if viewport.size == 0:
        return None
    # Integer RGB is interpreted in its dtype's native range. Floating RGB may
    # be normalized or 0..255. This avoids misclassifying a dark uint8 image
    # filled with value 1 as normalized white.
    if np.issubdtype(viewport.dtype, np.integer):
        threshold = float(pixel_threshold) * float(np.iinfo(viewport.dtype).max)
    else:
        maximum = float(np.max(viewport))
        threshold = float(pixel_threshold) * 255.0 if maximum > 1.5 else float(pixel_threshold)
    white_fraction = float(np.mean(viewport >= threshold))
    return white_fraction >= float(fraction_threshold)


class EpisodeTelemetryWrapper(gym.Wrapper):
    """Attach reporting-only per-view and episode metrics to copied info.

    The wrapper never changes an observation, reward, action mask, termination
    flag, or truncation flag.  It copies ``info`` before adding metrics.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        analysis_provider: Callable[[str], SlideAnalysis | None] | None = None,
        action_names: Sequence[str] = DEFAULT_ACTION_NAMES,
        navigation_actions: Iterable[int] = range(9),
        white_pixel_threshold: float = 0.98,
        white_fraction_threshold: float = 0.99,
    ) -> None:
        super().__init__(env)
        self.analysis_provider = analysis_provider
        self.action_names = tuple(str(name) for name in action_names)
        self.navigation_actions = frozenset(int(value) for value in navigation_actions)
        self.white_pixel_threshold = float(white_pixel_threshold)
        self.white_fraction_threshold = float(white_fraction_threshold)
        if not 0.0 <= self.white_pixel_threshold <= 1.0:
            raise ValueError("white_pixel_threshold must lie in [0,1]")
        if not 0.0 <= self.white_fraction_threshold <= 1.0:
            raise ValueError("white_fraction_threshold must lie in [0,1]")
        self._analysis: SlideAnalysis | None = None
        self._slide_id: str | None = None
        self._label: int | None = None
        self._action_counts: Counter[str] = Counter()
        self._magnifications: Counter[str] = Counter()
        self._visited_cells: set[int] = set()
        self._visited_bounds: list[tuple[int, int, int, int]] = []
        self._revisit_count = 0
        self._observed_views = 0
        self._white_views = 0
        self._white_observed_views = 0
        self._tissue_views = 0
        self._tissue_observed_views = 0
        self._tumor_views = 0
        self._tumor_cells: set[int] = set()
        self._first_tumor_step: int | None = None
        self._current_view_key: tuple[Any, ...] | None = None
        self._current_view_report: dict[str, Any] = {}

    def _action_name(self, action: int) -> str:
        return (
            self.action_names[action]
            if 0 <= action < len(self.action_names)
            else f"action_{action}"
        )

    @staticmethod
    def _canonical_bounds(info: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
        raw = info.get("canonical_cell_bounds_level0", info.get("canonical_bounds"))
        if raw is None:
            return None
        values = tuple(int(value) for value in raw)
        return values if len(values) == 4 else None

    def _view_key(self, info: Mapping[str, Any]) -> tuple[Any, ...]:
        return (info.get("slide_id"), info.get("cell_id"), self._canonical_bounds(info))

    def _measure_view(
        self,
        observation: Mapping[str, Any] | Any,
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        bounds = self._canonical_bounds(info)
        cell_id = info.get("cell_id")
        report = {
            "white_viewport": is_white_viewport(
                observation,
                pixel_threshold=self.white_pixel_threshold,
                fraction_threshold=self.white_fraction_threshold,
            ),
            "tissue_fraction": (
                self._analysis.tissue_fraction(bounds)
                if self._analysis is not None and bounds is not None
                else None
            ),
            "tumor_fraction": (
                self._analysis.tumor_fraction(int(cell_id))
                if self._analysis is not None
                and self._analysis.annotation_available
                and cell_id is not None
                else None
            ),
        }
        self._current_view_key = self._view_key(info)
        self._current_view_report = report
        return report

    def _record_view(
        self,
        observation: Mapping[str, Any] | Any,
        info: Mapping[str, Any],
        *,
        step: int,
    ) -> None:
        report = self._measure_view(observation, info)
        cell_id_raw = info.get("cell_id")
        if cell_id_raw is None:
            return
        cell_id = int(cell_id_raw)
        if cell_id in self._visited_cells:
            self._revisit_count += 1
        self._visited_cells.add(cell_id)
        self._observed_views += 1
        magnification = info.get("magnification")
        if magnification is not None:
            self._magnifications[f"{float(magnification):g}"] += 1
        bounds = self._canonical_bounds(info)
        if bounds is not None:
            self._visited_bounds.append(bounds)
        tissue_fraction = report["tissue_fraction"]
        if tissue_fraction is not None:
            self._tissue_observed_views += 1
            self._tissue_views += int(tissue_fraction > 0.0)
        white = report["white_viewport"]
        if white is not None:
            self._white_observed_views += 1
            self._white_views += int(white)
        if report["tumor_fraction"] is not None and float(report["tumor_fraction"]) > 0.0:
            self._tumor_views += 1
            self._tumor_cells.add(cell_id)
            if self._first_tumor_step is None:
                self._first_tumor_step = int(step)

    def _reporting_info(
        self,
        observation: Mapping[str, Any] | Any,
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Copy info and describe the current viewport without affecting behavior."""

        if self._current_view_key != self._view_key(info):
            self._measure_view(observation, info)
        enriched = dict(info)
        enriched.update(self._current_view_report)
        return enriched

    def _begin(self, observation: Mapping[str, Any] | Any, info: Mapping[str, Any]) -> None:
        self._slide_id = str(info.get("slide_id")) if info.get("slide_id") is not None else None
        self._label = int(info["label"]) if info.get("label") is not None else None
        self._analysis = (
            self.analysis_provider(self._slide_id)
            if self.analysis_provider is not None and self._slide_id is not None
            else None
        )
        self._action_counts.clear()
        self._magnifications.clear()
        self._visited_cells.clear()
        self._visited_bounds.clear()
        self._revisit_count = 0
        self._observed_views = 0
        self._white_views = 0
        self._white_observed_views = 0
        self._tissue_views = 0
        self._tissue_observed_views = 0
        self._tumor_views = 0
        self._tumor_cells.clear()
        self._first_tumor_step = None
        self._current_view_key = None
        self._current_view_report = {}
        self._record_view(observation, info, step=0)

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._begin(observation, info)
        return observation, _gymnasium_safe_info(self._reporting_info(observation, info))

    def _episode_metrics(
        self,
        info: Mapping[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        diagnosis = info.get("diagnosis", info.get("prediction"))
        if info.get("termination_reason") is not None:
            termination_reason = str(info["termination_reason"])
        elif truncated:
            termination_reason = "horizon"
        elif diagnosis is not None:
            termination_reason = "diagnosis"
        elif terminated:
            termination_reason = "terminated"
        else:  # pragma: no cover - called only for done transitions
            termination_reason = "unknown"
        tissue_coverage = (
            self._analysis.tissue_coverage(self._visited_bounds)
            if self._analysis is not None
            else None
        )
        tumor_coverage = (
            self._analysis.tumor_coverage(self._visited_bounds)
            if self._analysis is not None
            else None
        )
        annotation_available = bool(
            self._analysis is not None and self._analysis.annotation_available
        )
        annotation_exhaustive = bool(
            self._analysis is not None and self._analysis.annotation_exhaustive
        )
        return {
            "termination_reason": termination_reason,
            "classifier_source": info.get(
                "classifier_source", "policy_action" if diagnosis is not None else None
            ),
            "action_histogram": dict(sorted(self._action_counts.items())),
            "magnification_histogram": dict(sorted(self._magnifications.items())),
            "observed_views": self._observed_views,
            "unique_cells_visited": len(self._visited_cells),
            "revisit_count": self._revisit_count,
            "revisit_rate": (self._revisit_count / max(1, self._observed_views - 1)),
            "white_view_count": self._white_views,
            "white_observed_views": self._white_observed_views,
            "white_view_fraction": (
                self._white_views / self._white_observed_views
                if self._white_observed_views
                else None
            ),
            "tissue_coverage": tissue_coverage,
            "tissue_hit_count": (self._tissue_views if self._tissue_observed_views else None),
            "tissue_observed_views": self._tissue_observed_views,
            "tissue_hit_rate": (
                self._tissue_views / self._tissue_observed_views
                if self._tissue_observed_views
                else None
            ),
            "tumor_annotation_available": annotation_available,
            "tumor_annotation_exhaustive": annotation_exhaustive,
            "tumor_hit": bool(self._tumor_views) if annotation_available else None,
            "tumor_view_count": self._tumor_views if annotation_available else None,
            "tumor_view_fraction": (
                self._tumor_views / self._observed_views
                if annotation_available and self._observed_views
                else None
            ),
            "unique_tumor_cells_visited": (
                len(self._tumor_cells) if annotation_available else None
            ),
            "first_tumor_step": self._first_tumor_step if annotation_available else None,
            "tumor_coverage": tumor_coverage,
            "diagnosis_step": (
                int(info.get("actions_taken"))
                if diagnosis is not None and info.get("actions_taken") is not None
                else None
            ),
            "early_diagnosis": (
                bool(int(info.get("remaining_actions", 0)) > 0) if diagnosis is not None else False
            ),
        }

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        selected = int(action)
        self._action_counts[self._action_name(selected)] += 1
        if selected in self.navigation_actions and not bool(info.get("invalid_action", False)):
            step = int(info.get("actions_taken", sum(self._action_counts.values())))
            self._record_view(observation, info, step=step)
        enriched = self._reporting_info(observation, info)
        if not (terminated or truncated):
            return (
                observation,
                reward,
                terminated,
                truncated,
                _gymnasium_safe_info(enriched),
            )
        existing = enriched.get("episode_metrics")
        metrics = dict(existing) if isinstance(existing, Mapping) else {}
        metrics.update(
            self._episode_metrics(
                enriched,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )
        enriched["episode_metrics"] = metrics
        return (
            observation,
            reward,
            terminated,
            truncated,
            _gymnasium_safe_info(enriched),
        )


def episode_record_from_info(
    info: Mapping[str, Any],
    *,
    variant_id: str,
    global_step: int,
    update: int | None,
) -> dict[str, Any]:
    """Normalize one Gym terminal-info record into the study JSONL schema."""

    normalized_info = _without_gymnasium_presence_masks(info)
    metrics = normalized_info.get("episode_metrics")
    details = (
        {**normalized_info, **dict(metrics)} if isinstance(metrics, Mapping) else normalized_info
    )
    diagnosis = details.get("diagnosis", details.get("prediction"))
    label = details.get("label")
    correct = details.get("correct")
    if correct is None and diagnosis is not None and label is not None:
        correct = int(diagnosis) == int(label)
    names = (
        "slide_id",
        "start_position",
        "episode_return",
        "episode_length",
        "completed_episode_count",
        "completion_global_step",
        "termination_reason",
        "classifier_source",
        "belief",
        "final_belief",
        "evidence_count",
        "unique_evidence_count",
        "observed_views",
        "unique_cells_visited",
        "revisit_count",
        "revisit_rate",
        "inspected_fraction",
        "white_view_count",
        "white_observed_views",
        "white_view_fraction",
        "tissue_coverage",
        "tissue_hit_count",
        "tissue_observed_views",
        "tissue_hit_rate",
        "tumor_annotation_available",
        "tumor_annotation_exhaustive",
        "tumor_hit",
        "tumor_view_count",
        "tumor_view_fraction",
        "unique_tumor_cells_visited",
        "first_tumor_step",
        "tumor_coverage",
        "diagnosis_step",
        "early_diagnosis",
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "variant_id": str(variant_id),
        "global_step": int(global_step),
        "update": None if update is None else int(update),
        "episode_id": (
            int(details["completed_episode_count"])
            if details.get("completed_episode_count") is not None
            else None
        ),
        "label": None if label is None else int(label),
        "diagnosis": None if diagnosis is None else int(diagnosis),
        "correct": None if correct is None else bool(correct),
        "action_histogram": _named_count(details.get("action_histogram")),
        "magnification_histogram": _named_count(details.get("magnification_histogram")),
    }
    for name in names:
        if name in details:
            record[name] = _json_safe(details[name])
    if record.get("final_belief") is None and record.get("belief") is not None:
        record["final_belief"] = record["belief"]
    _validate_episode_histogram_totals(record)
    return record


def _mean(records: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [float(record[name]) for record in records if record.get(name) is not None]
    return float(np.mean(values)) if values else None


def aggregate_episode_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate exact counts and paper metrics without emitting non-finite values."""

    labels: list[int] = []
    predictions: list[int] = []
    for record in records:
        if record.get("label") is not None and record.get("diagnosis") is not None:
            labels.append(int(record["label"]))
            predictions.append(int(record["diagnosis"]))
    label_array = np.asarray(labels, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((label_array == 1) & (prediction_array == 1)))
    tn = int(np.sum((label_array == 0) & (prediction_array == 0)))
    fp = int(np.sum((label_array == 0) & (prediction_array == 1)))
    fn = int(np.sum((label_array == 1) & (prediction_array == 0)))
    action_counts: Counter[str] = Counter()
    magnification_counts: Counter[str] = Counter()
    for record in records:
        action_counts.update(_named_count(record.get("action_histogram")))
        magnification_counts.update(_named_count(record.get("magnification_histogram")))
    action_total = sum(action_counts.values())
    magnification_total = sum(magnification_counts.values())
    exhaustive_tumor = [
        record
        for record in records
        if record.get("label") == 1
        and record.get("tumor_annotation_available") is True
        and record.get("tumor_annotation_exhaustive") is True
    ]
    termination_counts = Counter(
        str(record["termination_reason"])
        for record in records
        if record.get("termination_reason") is not None
    )
    classified = len(labels)
    result: dict[str, Any] = {
        "episodes": len(records),
        "classified_episodes": classified,
        "accuracy": float(np.mean(label_array == prediction_array)) if classified else None,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "mean_episode_return": _mean(records, "episode_return"),
        "mean_episode_length": _mean(records, "episode_length"),
        "mean_revisit_rate": _mean(records, "revisit_rate"),
        "mean_tissue_coverage": _mean(records, "tissue_coverage"),
        "mean_tissue_hit_rate": _mean(records, "tissue_hit_rate"),
        "mean_white_view_fraction": _mean(records, "white_view_fraction"),
        "mean_tumor_view_fraction": _mean(exhaustive_tumor, "tumor_view_fraction"),
        "mean_tumor_coverage": _mean(exhaustive_tumor, "tumor_coverage"),
        "tumor_hit_rate_exhaustive": (
            float(np.mean([bool(record.get("tumor_hit")) for record in exhaustive_tumor]))
            if exhaustive_tumor
            else None
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "action_fractions": {
            name: count / action_total for name, count in sorted(action_counts.items())
        }
        if action_total
        else {},
        "magnification_counts": dict(sorted(magnification_counts.items())),
        "magnification_fractions": {
            name: count / magnification_total
            for name, count in sorted(magnification_counts.items())
        }
        if magnification_total
        else {},
        "termination_counts": dict(sorted(termination_counts.items())),
        "early_diagnosis_rate": (
            float(np.mean([bool(record.get("early_diagnosis", False)) for record in records]))
            if records
            else None
        ),
        "mean_diagnosis_step": _mean(
            [record for record in records if record.get("diagnosis") is not None],
            "diagnosis_step",
        ),
        "termination_fractions": {
            name: count / len(records) for name, count in sorted(termination_counts.items())
        }
        if records
        else {},
    }
    return result


def _numeric_tree(prefix: str, value: Any):
    if isinstance(value, Mapping):
        for name, item in value.items():
            yield from _numeric_tree(f"{prefix}/{name}" if prefix else str(name), item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        yield prefix, float(value)


class TrainingTelemetry:
    """Append-only rollout/episode recorder with checkpoint-resume reconciliation."""

    def __init__(
        self,
        output_dir: str | Path,
        writer: Any | None,
        variant_id: str,
        *,
        trailing_window_episodes: int = 256,
        action_names: Sequence[str] = DEFAULT_ACTION_NAMES,
    ) -> None:
        if trailing_window_episodes <= 0:
            raise ValueError("trailing_window_episodes must be positive")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.variant_id = str(variant_id)
        self.trailing_window_episodes = int(trailing_window_episodes)
        self.action_names = tuple(str(name) for name in action_names)
        self.episodes_path = self.output_dir / "training_episodes.jsonl"
        self.rollouts_path = self.output_dir / "training_rollouts.jsonl"
        self.state_path = self.output_dir / "telemetry_state.json"
        self._episodes: list[dict[str, Any]] = []
        self._window: deque[dict[str, Any]] = deque(maxlen=self.trailing_window_episodes)
        self._episode_ids: set[int] = set()
        self._rollout_steps: set[int] = set()
        self._status = "in_progress"

    def reconcile(self, step: int, completed_episode_count: int | None = None) -> None:
        """Discard telemetry newer than the restored PPO checkpoint and reload state."""

        restored_step = int(step)
        episodes = [
            record
            for record in _load_jsonl(self.episodes_path)
            if int(record.get("completion_global_step", record.get("global_step", -1)))
            <= restored_step
            and (
                completed_episode_count is None
                or record.get("episode_id") is None
                or int(record["episode_id"]) <= int(completed_episode_count)
            )
        ]
        rollouts = [
            record
            for record in _load_jsonl(self.rollouts_path)
            if int(record.get("global_step", -1)) <= restored_step
        ]
        if self.episodes_path.exists():
            _rewrite_jsonl(self.episodes_path, episodes)
        if self.rollouts_path.exists():
            _rewrite_jsonl(self.rollouts_path, rollouts)
        self._episodes = episodes
        self._window = deque(
            episodes[-self.trailing_window_episodes :], maxlen=self.trailing_window_episodes
        )
        self._episode_ids = {
            int(record["episode_id"]) for record in episodes if record.get("episode_id") is not None
        }
        self._rollout_steps = {int(record["global_step"]) for record in rollouts}
        self._status = "in_progress"
        self._write_state(restored_step)

    def _actions(self, buffer: Any) -> np.ndarray:
        value = getattr(buffer, "actions", buffer)
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=np.int64).reshape(-1)

    def _rollout_histogram(self, buffer: Any) -> dict[str, int]:
        actions = self._actions(buffer)
        counts: Counter[str] = Counter()
        for action in actions:
            index = int(action)
            name = (
                self.action_names[index]
                if 0 <= index < len(self.action_names)
                else f"action_{index}"
            )
            counts[name] += 1
        return dict(sorted(counts.items()))

    def _write_scalars(
        self,
        *,
        step: int,
        rollout_counts: Mapping[str, int],
        new_episodes: Sequence[Mapping[str, Any]],
        cumulative: Mapping[str, Any],
        window: Mapping[str, Any],
    ) -> None:
        if self.writer is None:
            return
        for episode in new_episodes:
            completion_step = int(episode.get("completion_global_step", step))
            for name in (
                "correct",
                "revisit_rate",
                "tissue_coverage",
                "white_view_fraction",
                "tumor_hit",
                "tumor_view_fraction",
            ):
                value = episode.get(name)
                if value is not None:
                    self.writer.add_scalar(
                        f"training_paper/episode/{name}", float(value), completion_step
                    )
        rollout_total = sum(rollout_counts.values())
        if rollout_total:
            for name, count in rollout_counts.items():
                self.writer.add_scalar(
                    f"training_paper/rollout/action_fraction/{name}",
                    count / rollout_total,
                    step,
                )
        for prefix, aggregate in (("cumulative", cumulative), ("window_256", window)):
            for name, value in _numeric_tree(f"training_paper/{prefix}", aggregate):
                self.writer.add_scalar(name, value, step)

    def _write_state(self, step: int) -> None:
        _atomic_json(
            self.state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": self._status,
                "variant_id": self.variant_id,
                "global_step": int(step),
                "episodes": len(self._episodes),
                "last_episode_id": max(self._episode_ids) if self._episode_ids else None,
                "episode_jsonl": str(self.episodes_path),
                "rollout_jsonl": str(self.rollouts_path),
                "cumulative": aggregate_episode_records(self._episodes),
                "window": aggregate_episode_records(list(self._window)),
            },
        )

    def record_rollout(
        self,
        step: int,
        buffer: Any,
        completed_episodes: Sequence[Mapping[str, Any]],
        *,
        update: int | None = None,
    ) -> dict[str, Any]:
        """Record one completed collection after ``PPOTrainer.collect_rollout``."""

        global_step = int(step)
        records: list[dict[str, Any]] = []
        for info in completed_episodes:
            record = episode_record_from_info(
                info,
                variant_id=self.variant_id,
                global_step=global_step,
                update=update,
            )
            episode_id = record.get("episode_id")
            if episode_id is not None and int(episode_id) in self._episode_ids:
                continue
            records.append(record)
            if episode_id is not None:
                self._episode_ids.add(int(episode_id))
        _append_jsonl(self.episodes_path, records)
        self._episodes.extend(records)
        self._window.extend(records)
        rollout_counts = self._rollout_histogram(buffer)
        rollout_total = sum(rollout_counts.values())
        cumulative = aggregate_episode_records(self._episodes)
        window = aggregate_episode_records(list(self._window))
        rollout_record = {
            "schema_version": SCHEMA_VERSION,
            "variant_id": self.variant_id,
            "global_step": global_step,
            "update": None if update is None else int(update),
            "transitions": rollout_total,
            "completed_episodes": len(records),
            "action_counts": rollout_counts,
            "action_fractions": {
                name: count / rollout_total for name, count in rollout_counts.items()
            }
            if rollout_total
            else {},
            "cumulative": cumulative,
            "window": window,
        }
        if global_step not in self._rollout_steps:
            _append_jsonl(self.rollouts_path, [rollout_record])
            self._rollout_steps.add(global_step)
        self._write_scalars(
            step=global_step,
            rollout_counts=rollout_counts,
            new_episodes=records,
            cumulative=cumulative,
            window=window,
        )
        self._write_state(global_step)
        return rollout_record

    def finalize(self, step: int) -> dict[str, Any]:
        """Mark telemetry complete after the final PPO checkpoint is durable."""

        self._status = "complete"
        self._write_state(int(step))
        return json.loads(self.state_path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_ACTION_NAMES",
    "EpisodeTelemetryWrapper",
    "TrainingTelemetry",
    "aggregate_episode_records",
    "episode_record_from_info",
    "is_white_viewport",
]
