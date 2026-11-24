# ruff: noqa: UP006, UP007, UP031, UP045
"""Shared long-run episode plans, compact traces, and checkpoint accounting.

This module deliberately uses Python 3.8-compatible syntax because the same
file is imported by the pinned HistoGym environment and the WSI-Lab
environment.  It contains no environment-specific behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import psutil

PLAN_SCHEMA_VERSION = 2
TRACE_SCHEMA_VERSION = 1
ACTION_CODE_TO_CLASS = {"P": "pan", "I": "zoom_in", "O": "zoom_out"}
ACTION_CLASS_TO_CODE = {value: key for key, value in ACTION_CODE_TO_CLASS.items()}
REQUESTED_CLASS_TO_ID = {"pan": 0, "zoom_in": 1, "zoom_out": 2}
REQUESTED_ID_TO_CLASS = {value: key for key, value in REQUESTED_CLASS_TO_ID.items()}
IO_FIELDS = (
    "read_bytes",
    "write_bytes",
    "read_chars",
    "write_chars",
    "read_syscalls",
    "write_syscalls",
)


class LongRunProtocolError(RuntimeError):
    """Raised when a long-run artifact does not exactly match the protocol."""


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def document_content_sha256(document: Mapping[str, Any]) -> str:
    """Hash a self-describing JSON document without its self-hash field."""
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LongRunProtocolError("cannot load JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise LongRunProtocolError("expected a JSON object in %s" % path)
    return value


def decode_priority_codes(codes: str) -> Tuple[str, ...]:
    if not isinstance(codes, str):
        raise LongRunProtocolError("priority_codes must be a string")
    try:
        return tuple(ACTION_CODE_TO_CLASS[code] for code in codes)
    except KeyError as exc:
        raise LongRunProtocolError("unknown abstract action code %r" % exc.args[0]) from exc


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    slide_ordinal: int
    slide_id: str
    priority_seed: int
    priority_codes: str

    @property
    def priorities(self) -> Tuple[str, ...]:
        return decode_priority_codes(self.priority_codes)


@dataclass(frozen=True)
class EpisodePlan:
    path: Path
    file_sha256: str
    content_sha256: str
    protocol_sha256: str
    manifest_sha256: str
    scenario: str
    checkpoints: Tuple[int, ...]
    episodes: Tuple[EpisodeSpec, ...]
    accepted_steps_per_episode: int
    raw: Mapping[str, Any]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def unique_slide_ids(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(episode.slide_id for episode in self.episodes))

    def __getitem__(self, episode_index: int) -> EpisodeSpec:
        return self.episodes[episode_index]


def _manifest_rows(manifest: Mapping[str, Any], protocol: Mapping[str, Any]) -> list:
    manifest_config = protocol.get("manifest")
    if not isinstance(manifest_config, dict):
        raise LongRunProtocolError("protocol.manifest is missing")
    key = manifest_config.get("cohort_key")
    if not isinstance(key, str):
        raise LongRunProtocolError("protocol manifest cohort_key must be a string")
    rows = manifest.get(key)
    expected = manifest_config.get("expected_slides")
    if not isinstance(rows, list) or len(rows) != expected:
        raise LongRunProtocolError(
            "manifest %r must contain exactly %r rows" % (key, expected)
        )
    ordinals = [row.get("ordinal") for row in rows if isinstance(row, dict)]
    slide_ids = [row.get("slide_id") for row in rows if isinstance(row, dict)]
    if ordinals != list(range(len(rows))):
        raise LongRunProtocolError("manifest rows must have consecutive fixed ordinals")
    if len(set(slide_ids)) != len(rows) or any(not isinstance(item, str) for item in slide_ids):
        raise LongRunProtocolError("manifest slide IDs must be unique strings")
    return rows


def _priority_codes(protocol: Mapping[str, Any], episode_index: int) -> Tuple[int, str]:
    rollout = protocol["rollout"]
    seed = int(rollout["action_priority_seed"]) + episode_index
    quotas = rollout["abstract_quotas"]
    classes = []
    for action_class in ("pan", "zoom_in", "zoom_out"):
        classes.extend([action_class] * int(quotas[action_class]))
    random.Random(seed).shuffle(classes)
    return seed, "".join(ACTION_CLASS_TO_CODE[item] for item in classes)


def _schedule_ordinals(
    scenario: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list:
    episode_count = int(scenario["episode_count"])
    kind = scenario["schedule_kind"]
    if kind == "balanced_random":
        repeats = int(scenario["episodes_per_slide"])
        ordinals = [int(row["ordinal"]) for row in rows for _ in range(repeats)]
        if len(ordinals) != episode_count:
            raise LongRunProtocolError("balanced schedule size does not match episode_count")
        random.Random(int(scenario["schedule_seed"])).shuffle(ordinals)
        return ordinals
    if kind == "single_slide":
        slide_id = scenario["single_slide_id"]
        matches = [int(row["ordinal"]) for row in rows if row["slide_id"] == slide_id]
        if len(matches) != 1:
            raise LongRunProtocolError("single slide %r is not unique in manifest" % slide_id)
        return [matches[0]] * episode_count
    raise LongRunProtocolError("unknown schedule kind %r" % kind)


def generate_episode_plan_document(
    protocol_path: Union[str, Path], manifest_path: Union[str, Path], scenario_name: str
) -> Dict[str, Any]:
    """Build one deterministic plan document without writing it."""
    protocol_path = Path(protocol_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    protocol = _load_json(protocol_path)
    manifest = _load_json(manifest_path)
    if protocol.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise LongRunProtocolError("long-run protocol schema must be 2")
    protocol_sha256 = sha256_file(protocol_path)
    manifest_sha256 = sha256_file(manifest_path)
    expected_manifest_hash = protocol.get("manifest", {}).get("sha256")
    if manifest_sha256 != expected_manifest_hash:
        raise LongRunProtocolError(
            "manifest hash drift: %s != %s" % (manifest_sha256, expected_manifest_hash)
        )
    scenarios = protocol.get("scenarios")
    if not isinstance(scenarios, dict) or scenario_name not in scenarios:
        raise LongRunProtocolError("unknown scenario %r" % scenario_name)
    scenario = scenarios[scenario_name]
    rows = _manifest_rows(manifest, protocol)
    by_ordinal = {int(row["ordinal"]): row for row in rows}
    ordinals = _schedule_ordinals(scenario, rows)
    episodes = []
    seen_priority_codes = set()
    for episode_index, ordinal in enumerate(ordinals):
        priority_seed, codes = _priority_codes(protocol, episode_index)
        if codes in seen_priority_codes:
            raise LongRunProtocolError(
                "duplicate action-priority order at episode %d" % episode_index
            )
        seen_priority_codes.add(codes)
        row = by_ordinal[ordinal]
        episodes.append(
            {
                "episode_index": episode_index,
                "slide_ordinal": ordinal,
                "slide_id": row["slide_id"],
                "priority_seed": priority_seed,
                "priority_codes": codes,
            }
        )
    adjacent_repeats = sum(
        left == right for left, right in zip(ordinals, ordinals[1:])
    )
    maximum_run = 0
    current_run = 0
    previous = None
    for ordinal in ordinals:
        current_run = current_run + 1 if ordinal == previous else 1
        maximum_run = max(maximum_run, current_run)
        previous = ordinal
    document = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "artifact_type": "histogym_comparison_episode_plan",
        "scenario": scenario_name,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "episode_count": int(scenario["episode_count"]),
        "accepted_steps_per_episode": int(
            protocol["rollout"]["accepted_steps_per_episode"]
        ),
        "checkpoints": list(scenario["checkpoints"]),
        "schedule": dict(scenario),
        "action_code_legend": dict(protocol["rollout"]["abstract_action_codes"]),
        "abstract_quotas": dict(protocol["rollout"]["abstract_quotas"]),
        "diagnostics": {
            "unique_slide_count": len(set(ordinals)),
            "unique_priority_order_count": len(seen_priority_codes),
            "adjacent_same_slide_transitions": adjacent_repeats,
            "slide_change_transitions": max(0, len(ordinals) - 1 - adjacent_repeats),
            "maximum_same_slide_run": maximum_run,
        },
        "episodes": episodes,
    }
    document["content_sha256"] = document_content_sha256(document)
    return document


def validate_episode_plan_document(
    document: Mapping[str, Any],
    protocol_path: Union[str, Path],
    manifest_path: Union[str, Path],
) -> None:
    if document.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise LongRunProtocolError("episode plan schema mismatch")
    content_hash = document.get("content_sha256")
    if content_hash != document_content_sha256(document):
        raise LongRunProtocolError("episode plan content hash mismatch")
    scenario = document.get("scenario")
    if not isinstance(scenario, str):
        raise LongRunProtocolError("episode plan scenario is missing")
    expected = generate_episode_plan_document(protocol_path, manifest_path, scenario)
    if dict(document) != expected:
        raise LongRunProtocolError(
            "episode plan does not exactly replay the fixed protocol and manifest"
        )


def load_episode_plan(
    plan_path: Union[str, Path],
    protocol_path: Union[str, Path],
    manifest_path: Union[str, Path],
) -> EpisodePlan:
    path = Path(plan_path).resolve()
    document = _load_json(path)
    validate_episode_plan_document(document, protocol_path, manifest_path)
    episodes = tuple(
        EpisodeSpec(
            episode_index=int(row["episode_index"]),
            slide_ordinal=int(row["slide_ordinal"]),
            slide_id=str(row["slide_id"]),
            priority_seed=int(row["priority_seed"]),
            priority_codes=str(row["priority_codes"]),
        )
        for row in document["episodes"]
    )
    return EpisodePlan(
        path=path,
        file_sha256=sha256_file(path),
        content_sha256=str(document["content_sha256"]),
        protocol_sha256=str(document["protocol_sha256"]),
        manifest_sha256=str(document["manifest_sha256"]),
        scenario=str(document["scenario"]),
        checkpoints=tuple(int(value) for value in document["checkpoints"]),
        episodes=episodes,
        accepted_steps_per_episode=int(document["accepted_steps_per_episode"]),
        raw=document,
    )


def write_json_exclusive(path: Union[str, Path], value: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return destination


def generate_episode_plan_files(
    protocol_path: Union[str, Path],
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    protocol = _load_json(protocol_path)
    output_root = Path(output_dir).resolve()
    destinations = {
        scenario: output_root / (scenario + ".json")
        for scenario in protocol.get("scenarios", {})
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise LongRunProtocolError(
            "refusing to overwrite episode plan(s): %s"
            % ", ".join(str(path) for path in existing)
        )
    documents = {
        scenario: generate_episode_plan_document(protocol_path, manifest_path, scenario)
        for scenario in destinations
    }
    return {
        scenario: write_json_exclusive(destinations[scenario], document)
        for scenario, document in documents.items()
    }


@dataclass(frozen=True)
class _ProcessCounters:
    cpu_ns: int
    read_bytes: int
    write_bytes: int
    read_chars: int
    write_chars: int
    read_syscalls: int
    write_syscalls: int


def _zero_process_counters() -> _ProcessCounters:
    return _ProcessCounters(0, 0, 0, 0, 0, 0, 0)


def _read_process_counters(process: psutil.Process) -> _ProcessCounters:
    cpu = process.cpu_times()
    try:
        io = process.io_counters()
    except (psutil.AccessDenied, AttributeError):
        return _ProcessCounters(int((cpu.user + cpu.system) * 1e9), 0, 0, 0, 0, 0, 0)
    return _ProcessCounters(
        int((cpu.user + cpu.system) * 1e9),
        int(io.read_bytes),
        int(io.write_bytes),
        int(io.read_chars),
        int(io.write_chars),
        int(io.read_count),
        int(io.write_count),
    )


class ProcessTreeBoundarySampler:
    """Low-cost cumulative process-tree snapshots at episode boundaries."""

    def __init__(self) -> None:
        self.root = psutil.Process(os.getpid())
        self._initial = {}  # type: Dict[Tuple[int, float], _ProcessCounters]
        self._maximum = {}  # type: Dict[Tuple[int, float], _ProcessCounters]
        self._baseline_rss = 0
        self._peak_rss = 0
        self._initialize()

    def _processes(self) -> list:
        try:
            return [self.root, *self.root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    @staticmethod
    def _key(process: psutil.Process) -> Tuple[int, float]:
        return process.pid, process.create_time()

    @staticmethod
    def _maximum_counter(left: _ProcessCounters, right: _ProcessCounters) -> _ProcessCounters:
        left_values = tuple(getattr(left, field) for field in left.__dataclass_fields__)
        right_values = tuple(getattr(right, field) for field in right.__dataclass_fields__)
        return _ProcessCounters(*(max(a, b) for a, b in zip(left_values, right_values)))

    def _capture(self, initialize: bool = False) -> Tuple[int, _ProcessCounters]:
        current_rss = 0
        for process in self._processes():
            try:
                key = self._key(process)
                counters = _read_process_counters(process)
                current_rss += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                continue
            if initialize:
                self._initial[key] = counters
            elif key not in self._initial:
                self._initial[key] = _zero_process_counters()
            self._maximum[key] = self._maximum_counter(
                self._maximum.get(key, _zero_process_counters()), counters
            )
        self._peak_rss = max(self._peak_rss, current_rss)
        totals = [0] * 7
        for key, maximum in self._maximum.items():
            initial = self._initial.get(key, _zero_process_counters())
            maximum_values = tuple(
                getattr(maximum, field) for field in maximum.__dataclass_fields__
            )
            initial_values = tuple(
                getattr(initial, field) for field in initial.__dataclass_fields__
            )
            for index, (after, before) in enumerate(zip(maximum_values, initial_values)):
                totals[index] += max(0, int(after) - int(before))
        return current_rss, _ProcessCounters(*totals)

    def _initialize(self) -> None:
        current_rss, _ = self._capture(initialize=True)
        self._baseline_rss = current_rss
        self._peak_rss = current_rss

    def sample(self) -> Dict[str, Any]:
        started = time.perf_counter_ns()
        current_rss, counters = self._capture()
        return {
            "baseline_rss_bytes": self._baseline_rss,
            "current_rss_bytes": current_rss,
            "cumulative_peak_rss_bytes": self._peak_rss,
            "cumulative_incremental_peak_rss_bytes": max(
                0, self._peak_rss - self._baseline_rss
            ),
            "cumulative_process_tree_cpu_ns": counters.cpu_ns,
            "io": {
                "read_bytes": counters.read_bytes,
                "write_bytes": counters.write_bytes,
                "read_chars": counters.read_chars,
                "write_chars": counters.write_chars,
                "read_syscalls": counters.read_syscalls,
                "write_syscalls": counters.write_syscalls,
            },
            "sample_latency_ns": time.perf_counter_ns() - started,
        }


ATTEMPT_ARRAY_DTYPES: Dict[str, np.dtype] = {
    "attempt_priority_index": np.dtype("uint8"),
    "attempt_requested_class": np.dtype("uint8"),
    "attempt_concrete_action": np.dtype("int16"),
    "attempt_accepted": np.dtype("uint8"),
    "attempt_latency_ns": np.dtype("uint64"),
    "attempt_cpu_ns": np.dtype("uint64"),
    "attempt_selection_latency_ns": np.dtype("uint64"),
    "attempt_observation_bytes": np.dtype("uint64"),
    "attempt_state_before_primary": np.dtype("int64"),
    "attempt_state_before_x": np.dtype("float64"),
    "attempt_state_before_y": np.dtype("float64"),
    "attempt_state_before_level": np.dtype("float32"),
    "attempt_state_after_primary": np.dtype("int64"),
    "attempt_state_after_x": np.dtype("float64"),
    "attempt_state_after_y": np.dtype("float64"),
    "attempt_state_after_level": np.dtype("float32"),
}

EPISODE_ARRAY_DTYPES: Dict[str, np.dtype] = {
    "episode_slide_ordinal": np.dtype("uint16"),
    "episode_construction_count": np.dtype("uint16"),
    "episode_construction_latency_ns": np.dtype("uint64"),
    "episode_construction_cpu_ns": np.dtype("uint64"),
    "episode_close_count": np.dtype("uint16"),
    "episode_close_latency_ns": np.dtype("uint64"),
    "episode_close_cpu_ns": np.dtype("uint64"),
    "episode_reset_latency_ns": np.dtype("uint64"),
    "episode_reset_cpu_ns": np.dtype("uint64"),
    "episode_reset_observation_bytes": np.dtype("uint64"),
    "episode_reset_through_rollout_wall_ns": np.dtype("uint64"),
    "episode_reset_through_rollout_cpu_ns": np.dtype("uint64"),
    "episode_observation_count": np.dtype("uint32"),
    "episode_observation_bytes": np.dtype("uint64"),
    "episode_postponement_events": np.dtype("uint32"),
    "episode_cumulative_online_wall_ns": np.dtype("uint64"),
    "episode_cumulative_online_cpu_ns": np.dtype("uint64"),
    "episode_baseline_rss_bytes": np.dtype("uint64"),
    "episode_current_rss_bytes": np.dtype("uint64"),
    "episode_cumulative_peak_rss_bytes": np.dtype("uint64"),
    "episode_cumulative_incremental_peak_rss_bytes": np.dtype("uint64"),
    "episode_cumulative_process_tree_cpu_ns": np.dtype("uint64"),
    "episode_resource_sample_latency_ns": np.dtype("uint64"),
    **{
        "episode_cumulative_io_" + field: np.dtype("uint64")
        for field in IO_FIELDS
    },
}

FINISH_EPISODE_METRIC_KEYS = (
    "construction_count",
    "construction_latency_ns",
    "construction_cpu_ns",
    "close_count",
    "close_latency_ns",
    "close_cpu_ns",
    "reset_latency_ns",
    "reset_cpu_ns",
    "reset_observation_bytes",
    "reset_through_rollout_wall_ns",
    "reset_through_rollout_cpu_ns",
    "observation_count",
    "observation_bytes",
    "postponement_events",
)


def _filled_array(size: int, dtype: np.dtype) -> np.ndarray:
    if dtype.kind == "f":
        return np.full(size, np.nan, dtype=dtype)
    if dtype.kind == "i":
        return np.full(size, -1, dtype=dtype)
    if dtype.kind == "u" and dtype.itemsize == 1:
        return np.full(size, 255, dtype=dtype)
    return np.zeros(size, dtype=dtype)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise LongRunProtocolError("%s must be a non-negative integer" % name)
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LongRunProtocolError("%s must be a non-negative integer" % name) from exc
    if integer < 0 or integer != value:
        raise LongRunProtocolError("%s must be a non-negative integer" % name)
    return integer


def _normalize_state(state: Sequence[Union[int, float]]) -> Tuple[int, float, float, float]:
    if not isinstance(state, (tuple, list)) or len(state) != 4:
        raise LongRunProtocolError("native state must be (primary, x, y, level)")
    primary = int(state[0])
    if primary != state[0]:
        raise LongRunProtocolError("native state primary value must be an integer")
    return primary, float(state[1]), float(state[2]), float(state[3])


def _float_differs(left: float, right: float, tolerance: float = 1e-9) -> bool:
    if math.isnan(left) and math.isnan(right):
        return False
    if math.isnan(left) != math.isnan(right):
        return True
    return abs(left - right) > tolerance


def _states_differ(
    before: Tuple[int, float, float, float], after: Tuple[int, float, float, float]
) -> bool:
    return before[0] != after[0] or any(
        _float_differs(left, right)
        for left, right in zip(before[1:], after[1:])
    )


class CompactTraceRecorder:
    """Preallocated compact trace with strict plan semantics.

    Allocate this object before starting resource monitoring.  The initial
    arrays are filled, rather than left as lazy ``empty`` pages, so the agreed
    102-attempt-per-episode reserve is faulted in before measurement.
    """

    def __init__(
        self,
        plan: EpisodePlan,
        backend: str,
        attempts_per_episode_capacity: int = 102,
    ) -> None:
        if backend not in ("wsi_lab", "histogym"):
            raise LongRunProtocolError("unknown backend %r" % backend)
        if attempts_per_episode_capacity < plan.accepted_steps_per_episode:
            raise LongRunProtocolError("attempt capacity cannot be smaller than accepted quota")
        self.plan = plan
        self.backend = backend
        self.initial_attempt_capacity = plan.episode_count * attempts_per_episode_capacity
        self._attempt_capacity = self.initial_attempt_capacity
        self._attempt_count = 0
        self._next_episode = 0
        self._active_episode = None  # type: Optional[int]
        self._active_priorities = None  # type: Optional[Tuple[str, ...]]
        self._finished_episode = None  # type: Optional[int]
        self._sealed_episodes = 0
        self._growth_events = []  # type: list
        self._attempt_arrays = {
            name: _filled_array(self._attempt_capacity, dtype)
            for name, dtype in ATTEMPT_ARRAY_DTYPES.items()
        }
        self._episode_arrays = {
            name: _filled_array(plan.episode_count, dtype)
            for name, dtype in EPISODE_ARRAY_DTYPES.items()
        }
        self._episode_arrays["episode_slide_ordinal"][:] = np.asarray(
            [episode.slide_ordinal for episode in plan.episodes], dtype=np.uint16
        )
        self._attempt_offsets = np.zeros(plan.episode_count + 1, dtype=np.uint64)

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    @property
    def capacity_growth_events(self) -> Tuple[Mapping[str, int], ...]:
        return tuple(self._growth_events)

    def _grow(self, required: int) -> None:
        old_capacity = self._attempt_capacity
        new_capacity = max(required, old_capacity + max(1024, old_capacity // 2))
        for name, old in tuple(self._attempt_arrays.items()):
            replacement = _filled_array(new_capacity, old.dtype)
            replacement[: self._attempt_count] = old[: self._attempt_count]
            self._attempt_arrays[name] = replacement
        self._attempt_capacity = new_capacity
        self._growth_events.append(
            {
                "at_attempt": self._attempt_count,
                "old_capacity": old_capacity,
                "new_capacity": new_capacity,
            }
        )

    def begin_episode(self, episode_index: int) -> EpisodeSpec:
        if self._active_episode is not None or self._finished_episode is not None:
            raise LongRunProtocolError("previous episode has not been finished and sealed")
        if episode_index != self._next_episode or episode_index >= self.plan.episode_count:
            raise LongRunProtocolError(
                "episodes must be recorded in plan order: expected %d, got %d"
                % (self._next_episode, episode_index)
            )
        self._active_episode = episode_index
        self._active_priorities = self.plan.episodes[episode_index].priorities
        self._attempt_offsets[episode_index] = self._attempt_count
        return self.plan.episodes[episode_index]

    def record_attempt(
        self,
        priority_index: int,
        requested_class: str,
        concrete_action: int,
        accepted: bool,
        latency_ns: int,
        cpu_ns: int,
        selection_latency_ns: int,
        observation_bytes: int,
        before_state: Sequence[Union[int, float]],
        after_state: Sequence[Union[int, float]],
    ) -> None:
        if self._active_episode is None:
            raise LongRunProtocolError("begin_episode must precede record_attempt")
        episode = self.plan.episodes[self._active_episode]
        priority_index = _nonnegative_integer(priority_index, "priority_index")
        if priority_index >= self.plan.accepted_steps_per_episode:
            raise LongRunProtocolError("priority_index is outside the episode plan")
        if self._active_priorities is None:  # defensive; active episode checked above
            raise LongRunProtocolError("active episode priorities are missing")
        expected_class = self._active_priorities[priority_index]
        if requested_class != expected_class:
            raise LongRunProtocolError(
                "attempt class %r does not match episode %d priority %d (%r)"
                % (requested_class, episode.episode_index, priority_index, expected_class)
            )
        if requested_class not in REQUESTED_CLASS_TO_ID:
            raise LongRunProtocolError("unknown requested class %r" % requested_class)
        if not isinstance(accepted, (bool, np.bool_)):
            raise LongRunProtocolError("accepted must be boolean")
        before = _normalize_state(before_state)
        after = _normalize_state(after_state)
        if bool(accepted) != _states_differ(before, after):
            raise LongRunProtocolError(
                "accepted flag must exactly match native state change"
            )
        if self._attempt_count >= self._attempt_capacity:
            self._grow(self._attempt_count + 1)
        index = self._attempt_count
        values = {
            "attempt_priority_index": priority_index,
            "attempt_requested_class": REQUESTED_CLASS_TO_ID[requested_class],
            "attempt_concrete_action": int(concrete_action),
            "attempt_accepted": int(bool(accepted)),
            "attempt_latency_ns": _nonnegative_integer(latency_ns, "latency_ns"),
            "attempt_cpu_ns": _nonnegative_integer(cpu_ns, "cpu_ns"),
            "attempt_selection_latency_ns": _nonnegative_integer(
                selection_latency_ns, "selection_latency_ns"
            ),
            "attempt_observation_bytes": _nonnegative_integer(
                observation_bytes, "observation_bytes"
            ),
            "attempt_state_before_primary": before[0],
            "attempt_state_before_x": before[1],
            "attempt_state_before_y": before[2],
            "attempt_state_before_level": before[3],
            "attempt_state_after_primary": after[0],
            "attempt_state_after_x": after[1],
            "attempt_state_after_y": after[2],
            "attempt_state_after_level": after[3],
        }
        for name, value in values.items():
            self._attempt_arrays[name][index] = value
        self._attempt_count += 1

    def _validate_active_episode_attempts(self, episode_index: int) -> None:
        start = int(self._attempt_offsets[episode_index])
        stop = self._attempt_count
        accepted_mask = self._attempt_arrays["attempt_accepted"][start:stop].astype(bool)
        accepted_priorities = self._attempt_arrays["attempt_priority_index"][start:stop][
            accepted_mask
        ]
        expected = np.arange(self.plan.accepted_steps_per_episode, dtype=np.uint8)
        if len(accepted_priorities) != len(expected) or not np.array_equal(
            np.sort(accepted_priorities), expected
        ):
            raise LongRunProtocolError(
                "episode %d did not accept every planned priority exactly once" % episode_index
            )
        accepted_classes = self._attempt_arrays["attempt_requested_class"][start:stop][
            accepted_mask
        ]
        expected_counts = self.plan.raw["abstract_quotas"]
        actual_counts = {
            action_class: int(np.count_nonzero(accepted_classes == action_id))
            for action_class, action_id in REQUESTED_CLASS_TO_ID.items()
        }
        if actual_counts != expected_counts:
            raise LongRunProtocolError(
                "episode %d quota mismatch: %r != %r"
                % (episode_index, actual_counts, expected_counts)
            )

    def finish_episode(self, metrics: Mapping[str, Any]) -> None:
        if self._active_episode is None:
            raise LongRunProtocolError("no active episode to finish")
        missing = set(FINISH_EPISODE_METRIC_KEYS) - set(metrics)
        extra = set(metrics) - set(FINISH_EPISODE_METRIC_KEYS)
        if missing or extra:
            raise LongRunProtocolError(
                "finish metrics mismatch; missing=%r extra=%r"
                % (sorted(missing), sorted(extra))
            )
        episode_index = self._active_episode
        self._validate_active_episode_attempts(episode_index)
        self._attempt_offsets[episode_index + 1] = self._attempt_count
        for key in FINISH_EPISODE_METRIC_KEYS:
            self._episode_arrays["episode_" + key][episode_index] = _nonnegative_integer(
                metrics[key], key
            )
        expected_observations = (
            int(metrics["reset_observation_bytes"] > 0)
            + int(self._attempt_offsets[episode_index + 1] - self._attempt_offsets[episode_index])
        )
        if int(metrics["observation_count"]) != expected_observations:
            raise LongRunProtocolError(
                "observation_count must include reset plus every native step result"
            )
        self._active_episode = None
        self._active_priorities = None
        self._finished_episode = episode_index

    def add_final_close(self, latency_ns: int, cpu_ns: int) -> None:
        if self._finished_episode != self.plan.episode_count - 1:
            raise LongRunProtocolError("final close can only be added to the last finished episode")
        index = self._finished_episode
        self._episode_arrays["episode_close_count"][index] += 1
        self._episode_arrays["episode_close_latency_ns"][index] += _nonnegative_integer(
            latency_ns, "latency_ns"
        )
        self._episode_arrays["episode_close_cpu_ns"][index] += _nonnegative_integer(
            cpu_ns, "cpu_ns"
        )

    def seal_episode_boundary(
        self,
        cumulative_online_wall_ns: int,
        cumulative_online_cpu_ns: int,
        resource_snapshot: Mapping[str, Any],
    ) -> None:
        if self._finished_episode is None:
            raise LongRunProtocolError("finish_episode must precede boundary sealing")
        index = self._finished_episode
        if index != self._sealed_episodes:
            raise LongRunProtocolError("episode boundaries must be sealed in order")
        required = {
            "baseline_rss_bytes",
            "current_rss_bytes",
            "cumulative_peak_rss_bytes",
            "cumulative_incremental_peak_rss_bytes",
            "cumulative_process_tree_cpu_ns",
            "io",
            "sample_latency_ns",
        }
        if set(resource_snapshot) != required:
            raise LongRunProtocolError("resource snapshot fields do not match shared schema")
        io = resource_snapshot["io"]
        if not isinstance(io, Mapping) or set(io) != set(IO_FIELDS):
            raise LongRunProtocolError("resource snapshot I/O fields do not match shared schema")
        cumulative_wall = _nonnegative_integer(
            cumulative_online_wall_ns, "cumulative_online_wall_ns"
        )
        cumulative_cpu = _nonnegative_integer(
            cumulative_online_cpu_ns, "cumulative_online_cpu_ns"
        )
        if index and cumulative_wall < int(
            self._episode_arrays["episode_cumulative_online_wall_ns"][index - 1]
        ):
            raise LongRunProtocolError("cumulative online wall time decreased")
        if index and cumulative_cpu < int(
            self._episode_arrays["episode_cumulative_online_cpu_ns"][index - 1]
        ):
            raise LongRunProtocolError("cumulative online CPU time decreased")
        self._episode_arrays["episode_cumulative_online_wall_ns"][index] = cumulative_wall
        self._episode_arrays["episode_cumulative_online_cpu_ns"][index] = cumulative_cpu
        for key in required - {"io"}:
            target = (
                "episode_resource_sample_latency_ns"
                if key == "sample_latency_ns"
                else "episode_" + key
            )
            self._episode_arrays[target][index] = _nonnegative_integer(
                resource_snapshot[key], key
            )
        for field in IO_FIELDS:
            self._episode_arrays["episode_cumulative_io_" + field][index] = (
                _nonnegative_integer(io[field], field)
            )
        self._sealed_episodes += 1
        self._next_episode += 1
        self._finished_episode = None

    def finalize(self) -> Dict[str, np.ndarray]:
        if self._active_episode is not None or self._finished_episode is not None:
            raise LongRunProtocolError("all episodes must be finished and sealed")
        if self._sealed_episodes != self.plan.episode_count:
            raise LongRunProtocolError(
                "trace has %d/%d sealed episodes"
                % (self._sealed_episodes, self.plan.episode_count)
            )
        arrays = {
            name: np.ascontiguousarray(value[: self._attempt_count])
            for name, value in self._attempt_arrays.items()
        }
        arrays.update(
            {name: np.ascontiguousarray(value) for name, value in self._episode_arrays.items()}
        )
        arrays["episode_attempt_offsets"] = np.ascontiguousarray(self._attempt_offsets)
        validate_trace_arrays(arrays, self.plan, self.backend)
        return arrays

    def write_compressed(
        self,
        trace_path: Union[str, Path],
        metadata_path: Union[str, Path],
        concrete_action_legend: Mapping[str, int],
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        arrays = self.finalize()
        trace_destination = Path(trace_path).resolve()
        metadata_destination = Path(metadata_path).resolve()
        if trace_destination.exists() or metadata_destination.exists():
            raise LongRunProtocolError("refusing to overwrite compact trace artifacts")
        trace_destination.parent.mkdir(parents=True, exist_ok=True)
        metadata_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = trace_destination.with_name(
            trace_destination.name + ".tmp-%d" % os.getpid()
        )
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
            os.link(str(temporary), str(trace_destination))
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
        metadata = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "artifact_type": "histogym_comparison_compact_online_trace",
            "backend": self.backend,
            "scenario": self.plan.scenario,
            "plan": {
                "path": str(self.plan.path),
                "file_sha256": self.plan.file_sha256,
                "content_sha256": self.plan.content_sha256,
                "protocol_sha256": self.plan.protocol_sha256,
                "manifest_sha256": self.plan.manifest_sha256,
            },
            "trace": {
                "path": str(trace_destination),
                "sha256": sha256_file(trace_destination),
                "episode_count": self.plan.episode_count,
                "attempt_count": self._attempt_count,
                "accepted_count": self.plan.episode_count
                * self.plan.accepted_steps_per_episode,
                "initial_attempt_capacity": self.initial_attempt_capacity,
                "final_attempt_capacity": self._attempt_capacity,
                "capacity_growth_events": list(self._growth_events),
            },
            "array_schema": {
                name: {"dtype": value.dtype.str, "shape": list(value.shape)}
                for name, value in sorted(arrays.items())
            },
            "requested_class_legend": dict(REQUESTED_CLASS_TO_ID),
            "concrete_action_legend": dict(concrete_action_legend),
            "native_state_schema": {
                "primary": "WSI-Lab cell_id or HistoGym DeepZoom depth",
                "x_y": "NaN for WSI-Lab or native HistoGym fractional coordinates",
                "level": "WSI-Lab magnification or NaN for HistoGym",
            },
            "extra": dict(extra_metadata or {}),
        }
        metadata["content_sha256"] = document_content_sha256(metadata)
        write_json_exclusive(metadata_destination, metadata)
        return metadata


def _expected_trace_keys() -> set:
    return set(ATTEMPT_ARRAY_DTYPES) | set(EPISODE_ARRAY_DTYPES) | {
        "episode_attempt_offsets"
    }


def _validate_array_schema(arrays: Mapping[str, np.ndarray], plan: EpisodePlan) -> None:
    if set(arrays) != _expected_trace_keys():
        raise LongRunProtocolError(
            "compact trace array keys mismatch; missing=%r extra=%r"
            % (
                sorted(_expected_trace_keys() - set(arrays)),
                sorted(set(arrays) - _expected_trace_keys()),
            )
        )
    attempt_count = len(arrays["attempt_accepted"])
    for name, dtype in ATTEMPT_ARRAY_DTYPES.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != (attempt_count,):
            raise LongRunProtocolError(
                "%s has dtype/shape %s/%r, expected %s/(%d,)"
                % (name, value.dtype, value.shape, dtype, attempt_count)
            )
    for name, dtype in EPISODE_ARRAY_DTYPES.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != (plan.episode_count,):
            raise LongRunProtocolError(
                "%s has dtype/shape %s/%r, expected %s/(%d,)"
                % (name, value.dtype, value.shape, dtype, plan.episode_count)
            )
    offsets = arrays["episode_attempt_offsets"]
    if offsets.dtype != np.dtype("uint64") or offsets.shape != (plan.episode_count + 1,):
        raise LongRunProtocolError("episode_attempt_offsets dtype/shape mismatch")
    if int(offsets[0]) != 0 or int(offsets[-1]) != attempt_count:
        raise LongRunProtocolError("episode attempt offsets do not span the trace")
    if np.any(offsets[1:] < offsets[:-1]):
        raise LongRunProtocolError("episode attempt offsets decreased")


def _validate_trace_lifecycle(
    arrays: Mapping[str, np.ndarray], plan: EpisodePlan, backend: str
) -> None:
    constructions = arrays["episode_construction_count"].astype(np.int64)
    closes = arrays["episode_close_count"].astype(np.int64)
    if backend == "wsi_lab":
        expected_constructions = np.zeros(plan.episode_count, dtype=np.int64)
        expected_constructions[0] = 1
        expected_closes = np.zeros(plan.episode_count, dtype=np.int64)
        expected_closes[-1] = 1
    elif backend == "histogym":
        expected_constructions = np.zeros(plan.episode_count, dtype=np.int64)
        expected_constructions[0] = 1
        for index in range(1, plan.episode_count):
            expected_constructions[index] = int(
                plan.episodes[index].slide_id != plan.episodes[index - 1].slide_id
            )
        # A slide change closes the prior environment immediately before the
        # new constructor; the final active environment closes at run end.
        expected_closes = expected_constructions.copy()
        expected_closes[0] = 0
        expected_closes[-1] += 1
    else:
        raise LongRunProtocolError("unknown trace backend %r" % backend)
    if not np.array_equal(constructions, expected_constructions):
        raise LongRunProtocolError("environment construction trace violates native lifecycle")
    if not np.array_equal(closes, expected_closes):
        raise LongRunProtocolError("environment close trace violates native lifecycle")


def validate_trace_arrays(
    arrays: Mapping[str, np.ndarray], plan: EpisodePlan, backend: str
) -> None:
    """Validate trace shape, every priority identity, quotas, and lifecycle."""
    _validate_array_schema(arrays, plan)
    expected_ordinals = np.asarray(
        [episode.slide_ordinal for episode in plan.episodes], dtype=np.uint16
    )
    if not np.array_equal(arrays["episode_slide_ordinal"], expected_ordinals):
        raise LongRunProtocolError("trace slide order does not match episode plan")
    offsets = arrays["episode_attempt_offsets"]
    total_accepted = 0
    for episode in plan.episodes:
        start = int(offsets[episode.episode_index])
        stop = int(offsets[episode.episode_index + 1])
        priorities = arrays["attempt_priority_index"][start:stop]
        requested = arrays["attempt_requested_class"][start:stop]
        accepted = arrays["attempt_accepted"][start:stop].astype(bool)
        expected_ids = np.asarray(
            [REQUESTED_CLASS_TO_ID[item] for item in episode.priorities], dtype=np.uint8
        )
        if np.any(priorities >= plan.accepted_steps_per_episode):
            raise LongRunProtocolError("trace contains an out-of-range priority index")
        if not np.array_equal(requested, expected_ids[priorities]):
            raise LongRunProtocolError(
                "episode %d attempt classes do not match priority identities"
                % episode.episode_index
            )
        accepted_priorities = priorities[accepted]
        if not np.array_equal(
            np.sort(accepted_priorities),
            np.arange(plan.accepted_steps_per_episode, dtype=np.uint8),
        ):
            raise LongRunProtocolError(
                "episode %d accepted priorities are incomplete or duplicated"
                % episode.episode_index
            )
        total_accepted += int(np.count_nonzero(accepted))
        if int(arrays["episode_observation_count"][episode.episode_index]) != stop - start + 1:
            raise LongRunProtocolError(
                "episode %d observation count must equal reset plus attempts"
                % episode.episode_index
            )
    if total_accepted != plan.episode_count * plan.accepted_steps_per_episode:
        raise LongRunProtocolError("compact trace accepted-step total mismatch")
    wall = arrays["episode_cumulative_online_wall_ns"]
    cpu = arrays["episode_cumulative_online_cpu_ns"]
    if np.any(wall[1:] < wall[:-1]) or np.any(cpu[1:] < cpu[:-1]):
        raise LongRunProtocolError("cumulative online time decreased")
    monotonic_resource_fields = [
        "episode_cumulative_peak_rss_bytes",
        "episode_cumulative_incremental_peak_rss_bytes",
        "episode_cumulative_process_tree_cpu_ns",
    ] + ["episode_cumulative_io_" + field for field in IO_FIELDS]
    for field in monotonic_resource_fields:
        if np.any(arrays[field][1:] < arrays[field][:-1]):
            raise LongRunProtocolError("cumulative resource field %s decreased" % field)
    _validate_trace_lifecycle(arrays, plan, backend)


def validate_trace_sidecar(
    trace_path: Union[str, Path],
    metadata_path: Union[str, Path],
    plan: EpisodePlan,
) -> Dict[str, np.ndarray]:
    metadata = _load_json(metadata_path)
    if metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise LongRunProtocolError("compact trace metadata schema mismatch")
    if metadata.get("content_sha256") != document_content_sha256(metadata):
        raise LongRunProtocolError("compact trace metadata content hash mismatch")
    plan_metadata = metadata.get("plan", {})
    expected_plan = {
        "file_sha256": plan.file_sha256,
        "content_sha256": plan.content_sha256,
        "protocol_sha256": plan.protocol_sha256,
        "manifest_sha256": plan.manifest_sha256,
    }
    for key, expected in expected_plan.items():
        if plan_metadata.get(key) != expected:
            raise LongRunProtocolError("compact trace plan %s hash mismatch" % key)
    trace_path = Path(trace_path).resolve()
    if metadata.get("trace", {}).get("sha256") != sha256_file(trace_path):
        raise LongRunProtocolError("compact NPZ file hash mismatch")
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise LongRunProtocolError("cannot read compact trace: %s" % exc) from exc
    backend = metadata.get("backend")
    if not isinstance(backend, str):
        raise LongRunProtocolError("compact trace backend is missing")
    validate_trace_arrays(arrays, plan, backend)
    actual_schema = {
        name: {"dtype": value.dtype.str, "shape": list(value.shape)}
        for name, value in sorted(arrays.items())
    }
    if metadata.get("array_schema") != actual_schema:
        raise LongRunProtocolError("compact trace metadata array schema mismatch")
    if metadata.get("scenario") != plan.scenario:
        raise LongRunProtocolError("compact trace scenario mismatch")
    return arrays


def _linear_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values.astype(np.float64), percentile))


def _latency_metrics(values: np.ndarray) -> Dict[str, Union[int, float]]:
    total = int(np.sum(values, dtype=np.uint64))
    count = int(values.size)
    return {
        "count": count,
        "total_ns": total,
        "mean_ns": float(total) / count if count else math.nan,
        "median_ns": _linear_percentile(values, 50.0),
        "p95_ns": _linear_percentile(values, 95.0),
        "min_ns": int(np.min(values)) if count else math.nan,
        "max_ns": int(np.max(values)) if count else math.nan,
    }


def build_checkpoint_rows(
    plan: EpisodePlan,
    arrays: Mapping[str, np.ndarray],
    backend: str,
    preparation_wall_ns: int = 0,
    preparation_cpu_ns: int = 0,
    import_wall_ns: int = 0,
    import_cpu_ns: int = 0,
    native_storage: Optional[Mapping[str, Any]] = None,
    harness_storage: Optional[Mapping[str, Any]] = None,
) -> list:
    """Return plot-ready cumulative rows at the immutable plan checkpoints."""
    validate_trace_arrays(arrays, plan, backend)
    preparation_wall_ns = _nonnegative_integer(preparation_wall_ns, "preparation_wall_ns")
    preparation_cpu_ns = _nonnegative_integer(preparation_cpu_ns, "preparation_cpu_ns")
    import_wall_ns = _nonnegative_integer(import_wall_ns, "import_wall_ns")
    import_cpu_ns = _nonnegative_integer(import_cpu_ns, "import_cpu_ns")
    offsets = arrays["episode_attempt_offsets"]
    rows = []
    for checkpoint in plan.checkpoints:
        episode_slice = slice(0, checkpoint)
        attempt_stop = int(offsets[checkpoint])
        attempt_slice = slice(0, attempt_stop)
        accepted_mask = arrays["attempt_accepted"][attempt_slice].astype(bool)
        accepted_steps = int(np.count_nonzero(accepted_mask))
        expected_accepted = checkpoint * plan.accepted_steps_per_episode
        if accepted_steps != expected_accepted:
            raise LongRunProtocolError("checkpoint accepted-step prefix mismatch")
        step_latencies = arrays["attempt_latency_ns"][attempt_slice]
        accepted_latencies = step_latencies[accepted_mask]
        reset_latencies = arrays["episode_reset_latency_ns"][episode_slice]
        construction_ns = int(
            np.sum(arrays["episode_construction_latency_ns"][episode_slice], dtype=np.uint64)
        )
        close_ns = int(
            np.sum(arrays["episode_close_latency_ns"][episode_slice], dtype=np.uint64)
        )
        reset_ns = int(np.sum(reset_latencies, dtype=np.uint64))
        step_ns = int(np.sum(step_latencies, dtype=np.uint64))
        native_component_ns = construction_ns + close_ns + reset_ns + step_ns
        online_wall_ns = int(
            arrays["episode_cumulative_online_wall_ns"][checkpoint - 1]
        )
        online_cpu_ns = int(arrays["episode_cumulative_online_cpu_ns"][checkpoint - 1])
        online_with_import_ns = import_wall_ns + online_wall_ns
        full_wall_ns = preparation_wall_ns + online_with_import_ns
        full_cpu_ns = preparation_cpu_ns + import_cpu_ns + online_cpu_ns
        attempts = attempt_stop
        no_ops = attempts - accepted_steps
        row = {
            "schema_version": 1,
            "backend": backend,
            "scenario": plan.scenario,
            "checkpoint_episodes": checkpoint,
            "accepted_steps": accepted_steps,
            "native_step_calls": attempts,
            "no_op_attempts": no_ops,
            "observation_count": int(
                np.sum(arrays["episode_observation_count"][episode_slice], dtype=np.uint64)
            ),
            "observation_bytes": int(
                np.sum(arrays["episode_observation_bytes"][episode_slice], dtype=np.uint64)
            ),
            "postponement_events": int(
                np.sum(arrays["episode_postponement_events"][episode_slice], dtype=np.uint64)
            ),
            "components_ns": {
                "preparation_wall": preparation_wall_ns,
                "import_wall": import_wall_ns,
                "environment_construction": construction_ns,
                "environment_close": close_ns,
                "reset": reset_ns,
                "all_native_step_calls": step_ns,
                "native_component_sum": native_component_ns,
                "cumulative_online_outer_wall": online_wall_ns,
                "online_with_import_wall": online_with_import_ns,
                "full_preparation_import_online_wall": full_wall_ns,
            },
            "components_cpu_ns": {
                "preparation": preparation_cpu_ns,
                "import": import_cpu_ns,
                "cumulative_online": online_cpu_ns,
                "full": full_cpu_ns,
            },
            "amortized": {
                "full_seconds_per_episode": full_wall_ns / 1e9 / checkpoint,
                "full_milliseconds_per_accepted_step": full_wall_ns
                / 1e6
                / accepted_steps,
                "online_with_import_seconds_per_episode": online_with_import_ns
                / 1e9
                / checkpoint,
                "online_with_import_milliseconds_per_accepted_step": online_with_import_ns
                / 1e6
                / accepted_steps,
                "online_lifecycle_seconds_per_episode": online_wall_ns
                / 1e9
                / checkpoint,
                "online_lifecycle_milliseconds_per_accepted_step": online_wall_ns
                / 1e6
                / accepted_steps,
            },
            "throughput": {
                "accepted_steps_per_second_step_calls": accepted_steps * 1e9 / step_ns
                if step_ns
                else math.nan,
                "accepted_steps_per_second_reset_plus_steps": accepted_steps
                * 1e9
                / (reset_ns + step_ns)
                if reset_ns + step_ns
                else math.nan,
                "accepted_steps_per_second_online_outer": accepted_steps
                * 1e9
                / online_wall_ns
                if online_wall_ns
                else math.nan,
            },
            "latency_ns": {
                "all_native_step_calls": _latency_metrics(step_latencies),
                "accepted_native_step_calls": _latency_metrics(accepted_latencies),
                "reset": _latency_metrics(reset_latencies),
            },
            "resource": {
                "baseline_rss_bytes": int(
                    arrays["episode_baseline_rss_bytes"][checkpoint - 1]
                ),
                "current_rss_bytes": int(
                    arrays["episode_current_rss_bytes"][checkpoint - 1]
                ),
                "cumulative_peak_rss_bytes": int(
                    arrays["episode_cumulative_peak_rss_bytes"][checkpoint - 1]
                ),
                "cumulative_incremental_peak_rss_bytes": int(
                    arrays["episode_cumulative_incremental_peak_rss_bytes"][checkpoint - 1]
                ),
                "cumulative_process_tree_cpu_ns": int(
                    arrays["episode_cumulative_process_tree_cpu_ns"][checkpoint - 1]
                ),
                "io": {
                    field: int(
                        arrays["episode_cumulative_io_" + field][checkpoint - 1]
                    )
                    for field in IO_FIELDS
                },
            },
            "storage": {
                "native_generated": dict(native_storage or {}),
                "harness_trace_and_reports": dict(harness_storage or {}),
                "sampled_post_run_and_not_in_online_timing": True,
            },
            "plan_content_sha256": plan.content_sha256,
        }
        rows.append(row)
    return rows
