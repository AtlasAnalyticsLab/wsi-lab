#!/usr/bin/env python3
"""Fresh preparation and compact long-run WSI-Lab measurements.

This is a private, ignored benchmark entrypoint.  It deliberately reuses the
native WSI-Lab preparation and environment construction paths from ``runner``
while leaving the original 100-slide/three-repetition command unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil

if __package__ in {None, ""}:  # pragma: no cover - operator convenience
    _REPOSITORY = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPOSITORY))

from benchmarks.histogym_comparison.common.longrun import (
    CompactTraceRecorder,
    EpisodePlan,
    LongRunProtocolError,
    ProcessTreeBoundarySampler,
    build_checkpoint_rows,
    load_episode_plan,
    validate_trace_sidecar,
)
from benchmarks.histogym_comparison.common.protocol import (
    EXPECTED_QUOTAS,
    directory_snapshot,
)
from benchmarks.histogym_comparison.common.resource_monitor import ResourceMonitor
from benchmarks.histogym_comparison.wsi.cohort import (
    CohortSelection,
    SlideSpec,
    atomic_json_dump,
    json_default,
    load_cohort,
)
from benchmarks.histogym_comparison.wsi.runner import (
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    RESULT_SCHEMA_VERSION,
    WorkspaceLayout,
    _assert_full_observation,
    _construct_environment,
    _load_native_context,
    _provenance,
    _verify_prepared_workspace,
    _verify_source_identity,
)
from benchmarks.histogym_comparison.wsi.runner import run_prepare as run_legacy_prepare
from benchmarks.histogym_comparison.wsi.scheduler import (
    AcceptedActionScheduler,
    abstract_class,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
HARNESS_ROOT = REPOSITORY_ROOT / "benchmarks/histogym_comparison"
DEFAULT_LONGRUN_PROTOCOL = HARNESS_ROOT / "longrun/protocol.json"
LONGRUN_RESULT_SCHEMA_VERSION = 2
ATTEMPTS_PER_EPISODE_CAPACITY = 102
MAX_ATTEMPTS_PER_EPISODE = 500


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _selection_for_plan(
    plan: EpisodePlan,
    manifest_path: str | Path,
) -> CohortSelection:
    """Select exactly the plan's unique slides in fixed manifest order."""

    full = load_cohort(manifest_path, cohort="measured", limit=None)
    if full.manifest_sha256 != plan.manifest_sha256:
        raise LongRunProtocolError("episode plan and cohort manifest hashes differ")
    by_ordinal = {slide.ordinal: slide for slide in full.slides}
    referenced_ids: set[str] = set()
    for episode in plan.episodes:
        slide = by_ordinal.get(episode.slide_ordinal)
        if slide is None or slide.slide_id != episode.slide_id:
            raise LongRunProtocolError(
                f"episode {episode.episode_index} slide ordinal/ID drifted from manifest"
            )
        referenced_ids.add(episode.slide_id)
    if referenced_ids != set(plan.unique_slide_ids):
        raise LongRunProtocolError("episode plan unique-slide diagnostics are inconsistent")
    slides = tuple(slide for slide in full.slides if slide.slide_id in referenced_ids)
    if len(slides) != len(referenced_ids):
        raise LongRunProtocolError("episode plan references a missing manifest slide")
    normalized = [slide.to_dict() for slide in slides]
    selection_sha256 = hashlib.sha256(_canonical_bytes(normalized)).hexdigest()
    return replace(
        full,
        slides=slides,
        selection_sha256=selection_sha256,
        limited=len(slides) != len(full.slides),
    )


def _plan_binding(plan: EpisodePlan, selection: CohortSelection) -> dict[str, Any]:
    return {
        "scenario": plan.scenario,
        "path": str(plan.path),
        "file_sha256": plan.file_sha256,
        "content_sha256": plan.content_sha256,
        "protocol_sha256": plan.protocol_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "selection_sha256": selection.selection_sha256,
        "episode_count": plan.episode_count,
        "accepted_steps_per_episode": plan.accepted_steps_per_episode,
        "unique_slide_ids": list(plan.unique_slide_ids),
    }


def _load_plan_and_selection(args: argparse.Namespace) -> tuple[EpisodePlan, CohortSelection]:
    plan = load_episode_plan(args.plan, args.longrun_protocol, args.manifest)
    selection = _selection_for_plan(plan, args.manifest)
    return plan, selection


def _attach_preparation_binding(
    result: dict[str, Any],
    layout: WorkspaceLayout,
    plan: EpisodePlan,
    selection: CohortSelection,
) -> None:
    """Bind fresh prepared state to one immutable scenario after measurement."""

    binding = _plan_binding(plan, selection)
    result["longrun_schema_version"] = LONGRUN_RESULT_SCHEMA_VERSION
    result["scenario"] = plan.scenario
    result["longrun_episode_plan"] = binding
    result["aggregate"]["planned_episode_count"] = plan.episode_count
    result["aggregate"]["planned_unique_slide_count"] = len(selection.slides)
    result["aggregate"]["online_repetitions"] = 1
    result_path = layout.results_root / "preparation.json"
    atomic_json_dump(result_path, result)
    marker = json.loads(layout.marker.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError("cannot bind an incomplete preparation workspace")
    marker["longrun_episode_plan"] = binding
    atomic_json_dump(layout.marker, marker)


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    plan, selection = _load_plan_and_selection(args)
    if args.resume:
        raise ValueError("long-run preparation must use a new workspace; --resume is forbidden")
    legacy_args = argparse.Namespace(
        config=args.config,
        manifest=args.manifest,
        cohort="measured",
        limit=None,
        monitor_interval=args.monitor_interval,
        gpu_index=args.gpu_index,
        workspace=args.workspace,
        device=args.device,
        resume=False,
        _selection_override=selection,
    )
    result = run_legacy_prepare(legacy_args)
    layout = WorkspaceLayout.from_root(args.workspace)
    _attach_preparation_binding(result, layout, plan, selection)
    return result


def _verify_longrun_preparation(
    marker: Mapping[str, Any],
    plan: EpisodePlan,
    selection: CohortSelection,
) -> None:
    observed = marker.get("longrun_episode_plan")
    expected = _plan_binding(plan, selection)
    if observed != expected:
        raise LongRunProtocolError(
            "prepared workspace is not bound to this exact long-run episode plan"
        )


def _preparation_cost(layout: WorkspaceLayout) -> tuple[int, int, dict[str, Any]]:
    result_path = layout.results_root / "preparation.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("phase") != "preparation":
        raise LongRunProtocolError("preparation result is incomplete")
    aggregate = result.get("aggregate", {})
    resources = result.get("resources", {})
    wall_ns = int(aggregate.get("total_preprocessing_ns", -1))
    if wall_ns < 0:
        raise LongRunProtocolError("preparation result has no total preprocessing time")
    cpu_ns = round(float(resources.get("cpu_seconds", 0.0)) * 1e9)
    native_storage = aggregate.get("backend_generated_storage")
    if not isinstance(native_storage, dict):
        raise LongRunProtocolError("preparation result has no native storage snapshot")
    return wall_ns, cpu_ns, dict(native_storage)


def _process_cpu_ns(process: psutil.Process) -> int:
    cpu = process.cpu_times()
    return int((cpu.user + cpu.system) * 1e9)


def _run_episode(
    *,
    env: Any,
    source: Any,
    recorder: CompactTraceRecorder,
    episode_index: int,
    slide: SlideSpec,
    construction_ns: int,
    construction_cpu_ns: int,
) -> dict[str, int]:
    spec = recorder.begin_episode(episode_index)
    if spec.slide_id != slide.slide_id or spec.slide_ordinal != slide.ordinal:
        raise LongRunProtocolError("native slide selection does not match episode plan")
    dummy_before = int(source.dummy_embedding_returns)
    reads_before = int(source.underlying_embedding_reads)
    episode_cpu_started = time.process_time_ns()
    episode_started = time.perf_counter_ns()
    reset_cpu_started = time.process_time_ns()
    reset_started = time.perf_counter_ns()
    observation, info = env.reset(
        seed=spec.priority_seed,
        options={"slide_id": spec.slide_id},
    )
    reset_latency_ns = time.perf_counter_ns() - reset_started
    reset_cpu_ns = time.process_time_ns() - reset_cpu_started
    reset_observation_bytes = _assert_full_observation(observation)
    if info.get("slide_id") != spec.slide_id or int(info.get("actions_taken", -1)) != 0:
        raise AssertionError(f"reset did not select a fresh episode for {spec.slide_id}")
    if float(info.get("belief", math.nan)) != 0.5:
        raise AssertionError("ConstantBelief must expose 0.5")

    scheduler = AcceptedActionScheduler(
        spec.slide_ordinal,
        initial_priorities=spec.priorities,
    )
    attempts = 0
    observation_count = 1
    observation_bytes = reset_observation_bytes
    postponement_events = 0
    while not scheduler.complete:
        attempts += 1
        if attempts > MAX_ATTEMPTS_PER_EPISODE:
            raise RuntimeError(f"too many no-op retries for episode {episode_index}")
        selection_started = time.perf_counter_ns()
        decision = scheduler.choose(info)
        selection_latency_ns = time.perf_counter_ns() - selection_started
        postponement_events += len(decision.postponed_items)
        before_cell = int(info["cell_id"])
        before_magnification = float(info["magnification"])
        before_actions = int(info["actions_taken"])
        step_cpu_started = time.process_time_ns()
        step_started = time.perf_counter_ns()
        next_observation, _reward, terminated, truncated, next_info = env.step(
            int(decision.concrete_action)
        )
        latency_ns = time.perf_counter_ns() - step_started
        cpu_ns = time.process_time_ns() - step_cpu_started
        next_observation_bytes = _assert_full_observation(next_observation)
        after_cell = int(next_info["cell_id"])
        after_magnification = float(next_info["magnification"])
        after_actions = int(next_info["actions_taken"])
        invalid = bool(next_info.get("invalid_action", False))
        # Native geometry assigns one non-overlapping global cell-ID interval
        # to every magnification, so every valid zoom necessarily changes ID.
        state_changed = after_cell != before_cell
        accepted = (not invalid) and state_changed and after_actions == before_actions + 1
        recorder.record_attempt(
            priority_index=decision.priority_index,
            requested_class=decision.requested_class,
            concrete_action=int(decision.concrete_action),
            accepted=accepted,
            latency_ns=latency_ns,
            cpu_ns=cpu_ns,
            selection_latency_ns=selection_latency_ns,
            observation_bytes=next_observation_bytes,
            before_state=(before_cell, math.nan, math.nan, before_magnification),
            after_state=(after_cell, math.nan, math.nan, after_magnification),
        )
        observation_count += 1
        observation_bytes += next_observation_bytes
        if terminated or truncated:
            raise AssertionError("navigation-only 100-step rollout terminated unexpectedly")
        if accepted:
            if abstract_class(decision.concrete_action) != decision.requested_class:
                raise AssertionError("concrete action class does not match planned class")
            scheduler.confirm(decision)
        else:
            scheduler.reject_no_op(decision, cell_id=before_cell)
        observation, info = next_observation, next_info

    scheduler.verify_complete()
    if source.underlying_embedding_reads != reads_before:
        raise AssertionError("online HDF5 feature reads changed during an episode")
    quotas = {name: scheduler.accepted_counts[name] for name in EXPECTED_QUOTAS}
    if quotas != EXPECTED_QUOTAS:
        raise AssertionError(f"episode quotas changed: {quotas}")
    reset_through_rollout_wall_ns = time.perf_counter_ns() - episode_started
    reset_through_rollout_cpu_ns = time.process_time_ns() - episode_cpu_started
    recorder.finish_episode(
        {
            "construction_count": int(episode_index == 0),
            "construction_latency_ns": construction_ns if episode_index == 0 else 0,
            "construction_cpu_ns": construction_cpu_ns if episode_index == 0 else 0,
            "close_count": 0,
            "close_latency_ns": 0,
            "close_cpu_ns": 0,
            "reset_latency_ns": reset_latency_ns,
            "reset_cpu_ns": reset_cpu_ns,
            "reset_observation_bytes": reset_observation_bytes,
            "reset_through_rollout_wall_ns": reset_through_rollout_wall_ns,
            "reset_through_rollout_cpu_ns": reset_through_rollout_cpu_ns,
            "observation_count": observation_count,
            "observation_bytes": observation_bytes,
            "postponement_events": postponement_events,
        }
    )
    return {
        "attempts": attempts,
        "dummy_embedding_returns": int(source.dummy_embedding_returns) - dummy_before,
        "underlying_embedding_reads": int(source.underlying_embedding_reads) - reads_before,
    }


def _action_legend() -> dict[str, int]:
    from wsi_lab.envs.actions import Action

    return {
        action.name.lower(): int(action)
        for action in Action
        if int(action) <= int(Action.ZOOM_OUT)
    }


def _write_checkpoint_table(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headings = (
        "backend",
        "scenario",
        "checkpoint_episodes",
        "accepted_steps",
        "native_step_calls",
        "no_op_attempts",
        "observation_count",
        "observation_bytes",
        "postponement_events",
        "full_seconds_per_episode",
        "full_milliseconds_per_accepted_step",
        "online_seconds_per_episode",
        "online_milliseconds_per_accepted_step",
        "accepted_steps_per_second_step_calls",
        "accepted_steps_per_second_reset_plus_steps",
        "accepted_steps_per_second_online_outer",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        import csv

        writer = csv.DictWriter(stream, fieldnames=headings)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "backend": row["backend"],
                    "scenario": row["scenario"],
                    "checkpoint_episodes": row["checkpoint_episodes"],
                    "accepted_steps": row["accepted_steps"],
                    "native_step_calls": row["native_step_calls"],
                    "no_op_attempts": row["no_op_attempts"],
                    "observation_count": row["observation_count"],
                    "observation_bytes": row["observation_bytes"],
                    "postponement_events": row["postponement_events"],
                    "full_seconds_per_episode": row["amortized"][
                        "full_seconds_per_episode"
                    ],
                    "full_milliseconds_per_accepted_step": row["amortized"][
                        "full_milliseconds_per_accepted_step"
                    ],
                    "online_seconds_per_episode": row["amortized"][
                        "online_lifecycle_seconds_per_episode"
                    ],
                    "online_milliseconds_per_accepted_step": row["amortized"][
                        "online_lifecycle_milliseconds_per_accepted_step"
                    ],
                    "accepted_steps_per_second_step_calls": row["throughput"][
                        "accepted_steps_per_second_step_calls"
                    ],
                    "accepted_steps_per_second_reset_plus_steps": row["throughput"][
                        "accepted_steps_per_second_reset_plus_steps"
                    ],
                    "accepted_steps_per_second_online_outer": row["throughput"][
                        "accepted_steps_per_second_online_outer"
                    ],
                }
            )


def run_online(args: argparse.Namespace) -> dict[str, Any]:
    operation_started = time.perf_counter_ns()
    process = psutil.Process(os.getpid())
    _verify_source_identity()
    plan, selection = _load_plan_and_selection(args)
    config_path = Path(args.config).expanduser().resolve()
    native_context_cpu_started = time.process_time_ns()
    native_context_started = time.perf_counter_ns()
    config, adapter = _load_native_context(config_path, selection)
    native_context_import_and_setup_wall_ns = (
        time.perf_counter_ns() - native_context_started
    )
    native_context_import_and_setup_cpu_ns = (
        time.process_time_ns() - native_context_cpu_started
    )
    layout = WorkspaceLayout.from_root(args.workspace)
    marker = _verify_prepared_workspace(layout, selection, config_path)
    _verify_longrun_preparation(marker, plan, selection)
    preparation_wall_ns, preparation_cpu_ns, native_storage = _preparation_cost(layout)
    result_path = layout.results_root / "longrun_online.json"
    trace_path = layout.raw_root / "longrun_online_trace.npz"
    trace_metadata_path = layout.raw_root / "longrun_online_trace.json"
    checkpoint_csv_path = layout.results_root / "longrun_checkpoints.csv"
    failure_path = layout.results_root / "longrun_online_failed.json"
    for path in (
        result_path,
        trace_path,
        trace_metadata_path,
        checkpoint_csv_path,
        failure_path,
    ):
        if path.exists():
            raise FileExistsError(f"long-run output already exists: {path}")

    recorder = CompactTraceRecorder(
        plan,
        "wsi_lab",
        attempts_per_episode_capacity=ATTEMPTS_PER_EPISODE_CAPACITY,
    )
    monitor = ResourceMonitor(
        interval_seconds=float(args.monitor_interval),
        gpu_index=None,
    )
    resources: dict[str, Any] | None = None
    boundary_sampler: ProcessTreeBoundarySampler | None = None
    env = None
    source = None
    try:
        monitor.start()
        boundary_sampler = ProcessTreeBoundarySampler()
        process_start_to_measurement_ns = max(
            0,
            time.time_ns() - int(process.create_time() * 1e9),
        )
        process_start_to_measurement_cpu_ns = _process_cpu_ns(process)
        online_cpu_started = time.process_time_ns()
        online_started = time.perf_counter_ns()
        construction_cpu_started = time.process_time_ns()
        construction_started = time.perf_counter_ns()
        env, source = _construct_environment(config, adapter, selection, layout)
        construction_ns = time.perf_counter_ns() - construction_started
        construction_cpu_ns = time.process_time_ns() - construction_cpu_started
        if env is None or source is None:  # defensive typing and fail-closed lifecycle
            raise AssertionError("environment construction returned no environment/source")
        active_env = env
        slides_by_id = {slide.slide_id: slide for slide in selection.slides}
        total_attempts = 0
        total_dummy_returns = 0
        monitor_samples_at_final_boundary = 0
        for episode_index, spec in enumerate(plan.episodes):
            episode_counts = _run_episode(
                env=active_env,
                source=source,
                recorder=recorder,
                episode_index=episode_index,
                slide=slides_by_id[spec.slide_id],
                construction_ns=construction_ns,
                construction_cpu_ns=construction_cpu_ns,
            )
            total_attempts += episode_counts["attempts"]
            total_dummy_returns += episode_counts["dummy_embedding_returns"]
            if episode_index == plan.episode_count - 1:
                close_cpu_started = time.process_time_ns()
                close_started = time.perf_counter_ns()
                active_env.close()
                close_ns = time.perf_counter_ns() - close_started
                close_cpu_ns = time.process_time_ns() - close_cpu_started
                env = None
                recorder.add_final_close(close_ns, close_cpu_ns)
            snapshot = boundary_sampler.sample()
            monitor_rss = monitor.prefix_rss_snapshot()
            # The boundary sampler supplies exact cumulative CPU/I/O and the
            # current RSS.  The asynchronous monitor supplies the within-step
            # RSS peak that boundary-only sampling could miss.
            prefix_peak = max(
                int(monitor_rss["cumulative_peak_rss_bytes"]),
                int(snapshot["cumulative_peak_rss_bytes"]),
                int(snapshot["current_rss_bytes"]),
            )
            prefix_baseline = int(monitor_rss["baseline_rss_bytes"])
            snapshot["baseline_rss_bytes"] = prefix_baseline
            snapshot["cumulative_peak_rss_bytes"] = prefix_peak
            snapshot["cumulative_incremental_peak_rss_bytes"] = max(
                0, prefix_peak - prefix_baseline
            )
            monitor_samples_at_final_boundary = int(monitor_rss["process_samples"])
            recorder.seal_episode_boundary(
                cumulative_online_wall_ns=time.perf_counter_ns() - online_started,
                cumulative_online_cpu_ns=time.process_time_ns() - online_cpu_started,
                resource_snapshot=snapshot,
            )
        if source is None:
            raise AssertionError("environment source was not constructed")
        if int(source.underlying_embedding_reads) != 0:
            raise AssertionError("online HDF5 feature reads must remain zero")
        if total_dummy_returns != int(source.dummy_embedding_returns):
            raise AssertionError("dummy embedding return accounting drifted")
        if total_attempts != recorder.attempt_count:
            raise AssertionError("compact trace attempt accounting drifted")
        resources = monitor.stop()

        serialization_started = time.perf_counter_ns()
        trace_metadata = recorder.write_compressed(
            trace_path,
            trace_metadata_path,
            _action_legend(),
            extra_metadata={
                "belief_model_loaded": False,
                "belief_checkpoint_loaded": False,
                "belief_value": 0.5,
                "online_embedding_reads": 0,
                "render_called": False,
                "diagnosis_actions": 0,
                "full_observation_generation": True,
                "lightweight_evaluation": False,
            },
        )
        trace_serialization_ns = time.perf_counter_ns() - serialization_started
        arrays = validate_trace_sidecar(trace_path, trace_metadata_path, plan)
        harness_storage = directory_snapshot(layout.raw_root)
        checkpoint_rows = build_checkpoint_rows(
            plan,
            arrays,
            "wsi_lab",
            preparation_wall_ns=preparation_wall_ns,
            preparation_cpu_ns=preparation_cpu_ns,
            # This explicit region contains native config/factory imports and
            # context setup. Further lazy imports remain inside construction.
            import_wall_ns=native_context_import_and_setup_wall_ns,
            import_cpu_ns=native_context_import_and_setup_cpu_ns,
            native_storage=native_storage,
            harness_storage=harness_storage,
        )
        _write_checkpoint_table(checkpoint_csv_path, checkpoint_rows)
        accepted_steps = plan.episode_count * plan.accepted_steps_per_episode
        final_checkpoint = checkpoint_rows[-1]
        result = {
            "schema_version": LONGRUN_RESULT_SCHEMA_VERSION,
            "legacy_result_schema_version": RESULT_SCHEMA_VERSION,
            "backend": "wsi-lab",
            "phase": "online_longrun",
            "scenario": plan.scenario,
            "repetition": 1,
            "status": "complete",
            "episode_plan": _plan_binding(plan, selection),
            "prepared_workspace": {
                "root": str(layout.root),
                "preparation_attempt": int(marker["attempt"]),
                "fresh_measurement_valid": bool(marker["fresh_measurement_valid"]),
                "cross_scenario_reuse": False,
            },
            "provenance": {
                **_provenance(config_path, selection),
                "longrun_protocol_path": str(Path(args.longrun_protocol).resolve()),
                "longrun_protocol_sha256": plan.protocol_sha256,
            },
            "aggregate": {
                "episode_count": plan.episode_count,
                "unique_slide_count": len(selection.slides),
                "accepted_steps": accepted_steps,
                "native_step_calls": total_attempts,
                "no_op_attempts": total_attempts - accepted_steps,
                "accepted_quotas": {
                    name: count * plan.episode_count
                    for name, count in EXPECTED_QUOTAS.items()
                },
                "environment_construction_count": 1,
                "environment_close_count": 1,
                "environment_construction_ns": int(
                    arrays["episode_construction_latency_ns"].sum(dtype="uint64")
                ),
                "environment_close_ns": int(
                    arrays["episode_close_latency_ns"].sum(dtype="uint64")
                ),
                "dummy_embedding_returns": total_dummy_returns,
                "underlying_embedding_reads": int(source.underlying_embedding_reads),
                "belief_model_loaded": False,
                "belief_checkpoint_loaded": False,
                "belief_value": 0.5,
                "render_called": False,
                "diagnosis_actions": 0,
                "full_observation_generation": True,
                "lightweight_evaluation": False,
                "process_start_to_measurement_ns": process_start_to_measurement_ns,
                "process_start_to_measurement_cpu_ns": process_start_to_measurement_cpu_ns,
                "resource_monitor_samples_at_final_episode_boundary": (
                    monitor_samples_at_final_boundary
                ),
                "native_context_import_and_setup_wall_ns": (
                    native_context_import_and_setup_wall_ns
                ),
                "native_context_import_and_setup_cpu_ns": (
                    native_context_import_and_setup_cpu_ns
                ),
                "module_import_accounting": (
                    "native config/factory import plus context setup is explicit; "
                    "further lazy imports remain inside environment_construction_ns"
                ),
                "trace_serialization_ns_excluded_from_measurement": trace_serialization_ns,
                "operation_wall_including_report_io_ns": (
                    time.perf_counter_ns() - operation_started
                ),
                "final_checkpoint": final_checkpoint,
            },
            "checkpoints": checkpoint_rows,
            "resources": resources,
            "storage": {
                "native_generated": native_storage,
                "harness_trace_and_reports_before_result": harness_storage,
                "sampled_after_monitoring": True,
            },
            "trace": trace_metadata,
            "plot_data": {
                "checkpoint_csv": str(checkpoint_csv_path),
                "checkpoint_json_embedded": True,
            },
        }
        atomic_json_dump(result_path, result)
        return result
    except BaseException as exc:
        if env is not None:
            with suppress(BaseException):
                env.close()
        if resources is None:
            with suppress(BaseException):
                resources = monitor.stop()
        failure = {
            "schema_version": LONGRUN_RESULT_SCHEMA_VERSION,
            "backend": "wsi-lab",
            "phase": "online_longrun",
            "scenario": plan.scenario,
            "repetition": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "resources": resources,
            "episode_plan": _plan_binding(plan, selection),
        }
        atomic_json_dump(failure_path, failure)
        raise


def _add_shared_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--longrun-protocol", type=Path, default=DEFAULT_LONGRUN_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--monitor-interval", type=float, default=0.1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Fresh plan-scoped WSI preparation.")
    _add_shared_inputs(prepare)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--gpu-index", type=int, default=0)
    prepare.add_argument("--resume", action="store_true")
    online = commands.add_parser("online", help="One compact long-run online pass.")
    _add_shared_inputs(online)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.monitor_interval <= 0:
        raise ValueError("--monitor-interval must be positive")
    if args.command == "prepare":
        result = run_prepare(args)
    elif args.command == "online":
        result = run_online(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    summary = {
        "backend": result["backend"],
        "phase": result["phase"],
        "scenario": result.get("scenario"),
        "status": result.get("status", "complete"),
        "aggregate": result["aggregate"],
    }
    print(json.dumps(summary, default=json_default, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
