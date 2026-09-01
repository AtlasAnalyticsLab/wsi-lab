#!/usr/bin/env python3
"""Long-run adapter for the pinned, unmodified HistoGym environment.

The adapter consumes an immutable shared episode plan.  One slide-bound native
``HistoEnv`` remains active while consecutive episodes use the same slide; a
slide change closes it and constructs the next one.  Native calls are timed
individually, while compact trace serialization happens after monitoring ends.

This file intentionally remains compatible with the pinned Python 3.8 runtime.
"""

from __future__ import print_function

import argparse
import contextlib
import csv
import datetime
import json
import os
import pathlib
import sys
import time
import traceback
from collections import Counter, deque


HERE = pathlib.Path(__file__).resolve().parent
COMPARISON_ROOT = HERE.parent
REPOSITORY_ROOT = COMPARISON_ROOT.parents[1]
if str(COMPARISON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARISON_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_histogym as legacy  # noqa: E402
from common import longrun as longrun_common  # noqa: E402
from common.resource_monitor import ResourceMonitor  # noqa: E402


DEFAULT_MANIFEST = COMPARISON_ROOT / "manifests" / "cohort.json"
DEFAULT_PROTOCOL = COMPARISON_ROOT / "longrun" / "protocol.json"
DEFAULT_UPSTREAM = HERE / "upstream" / "HistoGym"
EXPECTED_PREFIX = HERE / ".conda"
EXPECTED_PYTHON = "3.8.20"
EXPECTED_DEPENDENCIES = {
    "gym": "0.25.2",
    "numpy": "1.21.2",
    "Pillow": "9.5.0",
    "Shapely": "1.7.1",
    "matplotlib": "3.5.3",
    "opencv-python": "4.8.1.78",
    "openslide-python": "1.2.0",
    "psutil": "5.9.8",
    "openslide_library": "3.4.1",
}
RESULT_SCHEMA_VERSION = 2


class LongRunError(RuntimeError):
    pass


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_runtime(upstream):
    """Fail before opening a result or WSI if the pinned runtime drifted."""

    actual_prefix = pathlib.Path(sys.prefix).resolve()
    expected_prefix = EXPECTED_PREFIX.resolve()
    if actual_prefix != expected_prefix:
        raise LongRunError(
            "HistoGym long run must use %s, found %s"
            % (expected_prefix, actual_prefix)
        )
    if sys.version.split()[0] != EXPECTED_PYTHON:
        raise LongRunError(
            "HistoGym Python drift: %s != %s"
            % (sys.version.split()[0], EXPECTED_PYTHON)
        )
    dependencies = legacy.dependency_versions()
    mismatches = {
        name: {"expected": expected, "actual": dependencies.get(name)}
        for name, expected in EXPECTED_DEPENDENCIES.items()
        if dependencies.get(name) != expected
    }
    if mismatches:
        raise LongRunError(
            "HistoGym dependency drift: %s"
            % json.dumps(mismatches, sort_keys=True)
        )
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device.isdigit():
        raise LongRunError(
            "HistoGym wrapper must expose one configured numeric GPU index"
        )
    commit = legacy.verify_upstream(upstream)
    return {
        "python": sys.version,
        "dependencies": dependencies,
        "histogym_commit": commit,
        "pinned_environment_verified": True,
        "tracked_upstream_clean": True,
    }


def load_manifest_rows(path):
    """Use the legacy fail-closed reader and return rows indexed both ways."""

    _, rows = legacy.load_manifest(path, "measured", None)
    if len(rows) != 100:
        raise LongRunError("long-run measured manifest must contain exactly 100 slides")
    return rows, {row["slide_id"]: row for row in rows}


def expected_constructor_count(plan):
    """One constructor initially and one for each planned slide transition."""

    if not plan.episodes:
        return 0
    return 1 + sum(
        left.slide_id != right.slide_id
        for left, right in zip(plan.episodes, plan.episodes[1:])
    )


def should_close_before(plan, episode_index):
    """Return whether changing into this episode closes the prior slide."""

    return (
        episode_index > 0
        and plan[episode_index - 1].slide_id != plan[episode_index].slide_id
    )


def _native_result_path(output_dir, constructor_index, episode, slide_id):
    safe_slide = str(slide_id)
    if not legacy.SLIDE_ID_RE.match(safe_slide):
        raise LongRunError("unsafe slide ID %r" % safe_slide)
    return output_dir / "native_results" / (
        "%05d_episode_%05d_%s"
        % (constructor_index, episode.episode_index, safe_slide)
    )


def construct_environment(histogym, slide, native_result_path, quiet_stdout):
    arguments = (
        slide["wsi_path"],
        slide["xml_path"],
        legacy.TILE_SIZE,
        str(native_result_path),
    )
    return legacy.timed_call(
        histogym.HistoEnv,
        quiet_stdout,
        *arguments
    )


def close_environment(env, quiet_stdout):
    """Time native close plus the adapter's explicit OpenSlide-handle close."""

    total_wall_ns = 0
    total_cpu_ns = 0
    native_error = None
    try:
        _, elapsed_ns, cpu_ns = legacy.timed_call(env.close, quiet_stdout)
        total_wall_ns += int(elapsed_ns)
        total_cpu_ns += int(cpu_ns)
    except BaseException as exc:
        native_error = repr(exc)

    slide_handle = getattr(env, "slide", None)
    if slide_handle is not None and hasattr(slide_handle, "close"):
        cpu_started = time.process_time_ns()
        wall_started = time.perf_counter_ns()
        slide_handle.close()
        total_wall_ns += time.perf_counter_ns() - wall_started
        total_cpu_ns += time.process_time_ns() - cpu_started
    if native_error is not None:
        raise LongRunError("native HistoGym close failed: %s" % native_error)
    return total_wall_ns, total_cpu_ns


def _state_tuple(position_value):
    return (
        int(position_value[0]),
        float(position_value[1]),
        float(position_value[2]),
        float("nan"),
    )


def _new_pending(episode):
    return deque(
        {
            "priority_index": priority_index,
            "requested_class": requested_class,
            "postponement_count": 0,
            "unexpected_no_op_count": 0,
        }
        for priority_index, requested_class in enumerate(episode.priorities)
    )


def run_episode(env, episode, recorder, quiet_stdout):
    """Reset and execute one exact 100-accepted-action planned episode."""

    outer_wall_started = time.perf_counter_ns()
    outer_cpu_started = time.process_time_ns()
    raw_reset, reset_ns, reset_cpu_ns = legacy.timed_call(
        env.reset, quiet_stdout
    )
    reset_observation, reset_info = legacy.normalize_reset(raw_reset)
    del reset_info
    reset_descriptor = legacy.validate_observation(reset_observation)
    reset_depth = legacy.position(env)[0]
    native_max_depth = len(env.dz.level_tiles) - 1
    if native_max_depth < reset_depth:
        raise LongRunError("HistoGym native maximum depth precedes reset depth")

    pending = _new_pending(episode)
    pan_index = 0
    attempt_index = 0
    accepted_count = 0
    postponement_events = 0
    requested_counts = Counter()
    attempt_observation_bytes = 0
    step_latency_ns = 0
    step_cpu_ns = 0

    while pending:
        selected = None
        selected_actions = None
        selected_selection_ns = 0
        scan_length = len(pending)
        for _ in range(scan_length):
            item = pending.popleft()
            selection_started = time.perf_counter_ns()
            actions, _ = legacy.requested_class_viability(
                env, item["requested_class"], reset_depth, pan_index
            )
            selection_ns = time.perf_counter_ns() - selection_started
            if actions:
                selected = item
                selected_actions = actions
                selected_selection_ns = selection_ns
                break
            item["postponement_count"] += 1
            postponement_events += 1
            pending.append(item)
        if selected is None:
            raise LongRunError(
                "scheduler deadlock in episode %d at %r"
                % (episode.episode_index, legacy.position(env))
            )

        accepted_record = None
        for action in selected_actions:
            attempt = legacy.run_native_step(
                env,
                action,
                accepted_count,
                selected["priority_index"],
                selected["postponement_count"],
                attempt_index,
                selected["requested_class"],
                reset_depth,
                native_max_depth,
                quiet_stdout,
            )
            attempt["selection_latency_ns"] = selected_selection_ns
            selected_selection_ns = 0
            attempt_index += 1
            step_latency_ns += int(attempt["latency_ns"])
            step_cpu_ns += int(attempt["cpu_ns"])
            attempt_observation_bytes += int(attempt["observation_bytes"])
            recorder.record_attempt(
                priority_index=int(attempt["priority_index"]),
                requested_class=str(attempt["requested_class"]),
                concrete_action=int(attempt["concrete_action_code"]),
                accepted=bool(attempt["accepted"]),
                latency_ns=int(attempt["latency_ns"]),
                cpu_ns=int(attempt["cpu_ns"]),
                selection_latency_ns=int(attempt["selection_latency_ns"]),
                observation_bytes=int(attempt["observation_bytes"]),
                before_state=_state_tuple(attempt["position_before"]),
                after_state=_state_tuple(attempt["position_after"]),
            )
            if attempt["terminated"] or attempt["truncated"]:
                raise LongRunError(
                    "HistoGym terminated before 100 accepted actions in episode %d"
                    % episode.episode_index
                )
            if attempt["accepted"]:
                accepted_record = attempt
                break

        if accepted_record is None:
            selected["unexpected_no_op_count"] += 1
            selected["postponement_count"] += 1
            postponement_events += 1
            if selected["unexpected_no_op_count"] > 2:
                raise LongRunError(
                    "repeated unexpected no-op for episode %d priority %d"
                    % (episode.episode_index, selected["priority_index"])
                )
            pending.append(selected)
            continue

        accepted_count += 1
        requested_counts[selected["requested_class"]] += 1
        if selected["requested_class"] == "pan":
            pan_index += 1

    expected_quotas = dict(legacy.EXPECTED_QUOTAS)
    observed_quotas = {
        action_class: requested_counts[action_class]
        for action_class in expected_quotas
    }
    if accepted_count != 100 or observed_quotas != expected_quotas:
        raise LongRunError(
            "episode %d accepted protocol mismatch: %d, %r"
            % (episode.episode_index, accepted_count, observed_quotas)
        )
    return {
        "reset_latency_ns": int(reset_ns),
        "reset_cpu_ns": int(reset_cpu_ns),
        "reset_observation_bytes": int(reset_descriptor["bytes"]),
        "attempt_count": int(attempt_index),
        "accepted_count": int(accepted_count),
        "no_op_count": int(attempt_index - accepted_count),
        "postponement_events": int(postponement_events),
        "step_latency_ns": int(step_latency_ns),
        "step_cpu_ns": int(step_cpu_ns),
        "observation_count": int(1 + attempt_index),
        "observation_bytes": int(
            reset_descriptor["bytes"] + attempt_observation_bytes
        ),
        "reset_through_rollout_wall_ns": int(
            time.perf_counter_ns() - outer_wall_started
        ),
        "reset_through_rollout_cpu_ns": int(
            time.process_time_ns() - outer_cpu_started
        ),
    }


def _artifact_storage(output_dir):
    native = legacy.directory_usage(output_dir / "native_results")
    harness_paths = [
        output_dir / "trace.npz",
        output_dir / "trace.metadata.json",
        output_dir / "checkpoints.csv",
        output_dir / "result.json",
    ]
    harness_logical = 0
    harness_allocated = 0
    harness_files = 0
    for path in harness_paths:
        if path.is_file():
            stat = path.stat()
            harness_files += 1
            harness_logical += int(stat.st_size)
            harness_allocated += int(stat.st_blocks * 512)
    return {
        "native_generated": native,
        "harness_generated": {
            "file_count": harness_files,
            "logical_bytes": harness_logical,
            "allocated_bytes": harness_allocated,
            "excluded_from_native_storage": True,
        },
    }


def _native_storage_at_checkpoints(output_dir, plan):
    """Reconstruct exact native storage prefixes after monitoring stops.

    With rendering disabled, upstream HistoGym creates exactly one empty result
    directory per constructor and never writes into it.  The adapter-owned
    ``native_results`` parent is excluded from native storage.  This lets us
    stat each immutable constructor directory once after the measured phase and
    assign its exact allocated bytes to every cumulative checkpoint without a
    timed filesystem walk.
    """

    native_root = output_dir / "native_results"
    if native_root.is_symlink() or not native_root.is_dir():
        raise LongRunError("native result root is missing or not a directory")

    constructor_records = []
    constructor_index = 0
    for episode in plan.episodes:
        if episode.episode_index == 0 or should_close_before(
            plan, episode.episode_index
        ):
            path = _native_result_path(
                output_dir,
                constructor_index,
                episode,
                episode.slide_id,
            )
            constructor_records.append((episode.episode_index, path))
            constructor_index += 1

    expected_paths = {path for _, path in constructor_records}
    actual_paths = set(native_root.iterdir())
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        unexpected = sorted(str(path) for path in actual_paths - expected_paths)
        raise LongRunError(
            "native result directory set differs from constructors: "
            "missing=%r unexpected=%r" % (missing, unexpected)
        )
    parent_stat = native_root.stat()
    parent_diagnostic = {
        "path": str(native_root),
        "logical_bytes_final_post_run": int(parent_stat.st_size),
        "allocated_bytes_final_post_run": int(parent_stat.st_blocks * 512),
        "sampled_post_run": True,
        "excluded_from_exact_historical_native_prefix": True,
        "reason": (
            "adapter-owned parent directory allocation at earlier checkpoints "
            "cannot be reconstructed after all constructor entries exist"
        ),
    }

    allocated_by_path = {}
    for _, path in constructor_records:
        if path.is_symlink() or not path.is_dir():
            raise LongRunError("native constructor result is not a directory: %s" % path)
        entries = list(path.iterdir())
        if entries:
            raise LongRunError(
                "HistoGym native result directory is not empty with render disabled: %s"
                % path
            )
        allocated_by_path[path] = int(path.stat().st_blocks * 512)

    snapshots = {}
    for checkpoint in plan.checkpoints:
        prefix_paths = [
            path
            for episode_index, path in constructor_records
            if episode_index < checkpoint
        ]
        snapshots[int(checkpoint)] = {
            "root": str(native_root),
            "file_count": 0,
            "directory_count": len(prefix_paths),
            "symlink_count": 0,
            "logical_bytes": 0,
            "allocated_bytes": sum(allocated_by_path[path] for path in prefix_paths),
            "measurement": "posthoc_exact_empty_constructor_directories",
            "checkpoint_episodes": int(checkpoint),
            "adapter_owned_parent_directory_excluded": True,
            "adapter_owned_parent_directory_diagnostic": dict(parent_diagnostic),
            "render_disabled_and_directories_verified_empty": True,
        }
    return snapshots


def _write_checkpoint_table(path, rows):
    """Write the same compact plot-ready checkpoint table as WSI-Lab."""

    if path.exists():
        raise LongRunError("refusing to overwrite checkpoint table: %s" % path)
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


def run(args):
    operation_started_ns = time.perf_counter_ns()
    manifest_path = args.manifest.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    plan_path = args.episode_plan.expanduser().resolve()
    upstream = args.histogym_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == upstream or upstream in output_dir.parents:
        raise LongRunError("output cannot be inside the pinned HistoGym checkout")
    if output_dir.exists():
        raise LongRunError("fresh long-run output path must be absent: %s" % output_dir)

    runtime = validate_runtime(upstream)
    runtime["started_at_utc"] = utc_now()
    plan = longrun_common.load_episode_plan(
        plan_path, protocol_path, manifest_path
    )
    if args.scenario is not None and args.scenario != plan.scenario:
        raise LongRunError(
            "scenario argument %r differs from plan %r"
            % (args.scenario, plan.scenario)
        )
    rows, rows_by_id = load_manifest_rows(manifest_path)
    del rows
    for episode in plan.episodes:
        row = rows_by_id.get(episode.slide_id)
        if row is None or int(row["ordinal"]) != episode.slide_ordinal:
            raise LongRunError(
                "plan/manifest slide mismatch at episode %d"
                % episode.episode_index
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "native_results").mkdir()
    checkpoint_csv_path = output_dir / "checkpoints.csv"
    expected_constructors = expected_constructor_count(plan)

    import_cpu_started = time.process_time_ns()
    histogym, import_latency_ns = legacy.load_histogym(upstream)
    import_cpu_ns = time.process_time_ns() - import_cpu_started
    runtime.update(
        {
            "upstream_import_latency_ns": int(import_latency_ns),
            "upstream_import_cpu_ns": int(import_cpu_ns),
            "process_entry_to_import_complete_ns": int(
                time.perf_counter_ns() - operation_started_ns
            ),
        }
    )

    recorder = longrun_common.CompactTraceRecorder(
        plan,
        backend="histogym",
        attempts_per_episode_capacity=int(args.attempt_capacity),
    )
    resource_monitor = ResourceMonitor(
        interval_seconds=float(args.monitor_interval), gpu_index=None
    ).start()
    boundary_sampler = longrun_common.ProcessTreeBoundarySampler()
    online_wall_started = time.perf_counter_ns()
    online_cpu_started = time.process_time_ns()
    active_env = None
    active_slide_id = None
    constructor_count = 0
    total_postponements = 0
    total_no_ops = 0
    failed = None
    failure_traceback = None
    completed_episodes = 0
    monitor_samples_at_final_boundary = 0

    try:
        with open(os.devnull, "w") as quiet_stdout:
            for episode in plan.episodes:
                recorder.begin_episode(episode.episode_index)
                construction_count = 0
                construction_latency_ns = 0
                construction_cpu_ns = 0
                close_count = 0
                close_latency_ns = 0
                close_cpu_ns = 0
                if active_env is not None and should_close_before(
                    plan, episode.episode_index
                ):
                    close_latency_ns, close_cpu_ns = close_environment(
                        active_env, quiet_stdout
                    )
                    active_env = None
                    active_slide_id = None
                    close_count = 1
                if active_env is None:
                    slide = rows_by_id[episode.slide_id]
                    native_path = _native_result_path(
                        output_dir,
                        constructor_count,
                        episode,
                        episode.slide_id,
                    )
                    active_env, construction_latency_ns, construction_cpu_ns = (
                        construct_environment(
                            histogym, slide, native_path, quiet_stdout
                        )
                    )
                    active_slide_id = episode.slide_id
                    constructor_count += 1
                    construction_count = 1
                elif active_slide_id != episode.slide_id:
                    raise LongRunError(
                        "active HistoGym slide changed without a lifecycle boundary"
                    )

                metrics = run_episode(
                    active_env, episode, recorder, quiet_stdout
                )
                total_postponements += metrics["postponement_events"]
                total_no_ops += metrics["no_op_count"]

                recorder.finish_episode(
                    {
                        "construction_count": construction_count,
                        "construction_latency_ns": int(construction_latency_ns),
                        "construction_cpu_ns": int(construction_cpu_ns),
                        "close_count": close_count,
                        "close_latency_ns": int(close_latency_ns),
                        "close_cpu_ns": int(close_cpu_ns),
                        "reset_latency_ns": metrics["reset_latency_ns"],
                        "reset_cpu_ns": metrics["reset_cpu_ns"],
                        "reset_observation_bytes": metrics[
                            "reset_observation_bytes"
                        ],
                        "reset_through_rollout_wall_ns": metrics[
                            "reset_through_rollout_wall_ns"
                        ],
                        "reset_through_rollout_cpu_ns": metrics[
                            "reset_through_rollout_cpu_ns"
                        ],
                        "observation_count": metrics["observation_count"],
                        "observation_bytes": metrics["observation_bytes"],
                        "postponement_events": metrics["postponement_events"],
                    }
                )
                if episode.episode_index == plan.episode_count - 1:
                    close_latency_ns, close_cpu_ns = close_environment(
                        active_env, quiet_stdout
                    )
                    active_env = None
                    active_slide_id = None
                    recorder.add_final_close(close_latency_ns, close_cpu_ns)
                snapshot = boundary_sampler.sample()
                monitor_rss = resource_monitor.prefix_rss_snapshot()
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
                monitor_samples_at_final_boundary = int(
                    monitor_rss["process_samples"]
                )
                recorder.seal_episode_boundary(
                    cumulative_online_wall_ns=(
                        time.perf_counter_ns() - online_wall_started
                    ),
                    cumulative_online_cpu_ns=(
                        time.process_time_ns() - online_cpu_started
                    ),
                    resource_snapshot=snapshot,
                )
                completed = episode.episode_index + 1
                completed_episodes = completed
    except BaseException as exc:
        failed = exc
        failure_traceback = traceback.format_exc()
        if active_env is not None:
            with open(os.devnull, "w") as failure_quiet:
                with contextlib.suppress(BaseException):
                    close_environment(active_env, failure_quiet)
    monitored_resources = resource_monitor.stop()
    if failed is not None:
        failure = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "backend": "histogym",
            "scenario": plan.scenario,
            "status": "failed",
            "error_type": type(failed).__name__,
            "error": str(failed),
            "traceback": failure_traceback,
            "completed_episodes": completed_episodes,
            "resources": monitored_resources,
        }
        longrun_common.write_json_exclusive(
            output_dir / "failure.json", failure
        )
        raise failed

    if constructor_count != expected_constructors:
        raise LongRunError(
            "constructor count %d != planned %d"
            % (constructor_count, expected_constructors)
        )
    trace_metadata = recorder.write_compressed(
        output_dir / "trace.npz",
        output_dir / "trace.metadata.json",
        concrete_action_legend=legacy.ACTION_CODES,
        extra_metadata={
            "runtime": runtime,
            "expected_constructor_count": expected_constructors,
            "observed_constructor_count": constructor_count,
            "postponement_events": total_postponements,
            "no_op_attempts": total_no_ops,
            "runner_path": str(pathlib.Path(__file__).resolve()),
            "runner_sha256": longrun_common.sha256_file(__file__),
            "legacy_adapter_path": str(pathlib.Path(legacy.__file__).resolve()),
            "legacy_adapter_sha256": longrun_common.sha256_file(legacy.__file__),
        },
    )
    arrays = longrun_common.validate_trace_sidecar(
        output_dir / "trace.npz",
        output_dir / "trace.metadata.json",
        plan,
    )
    native_storage_by_checkpoint = _native_storage_at_checkpoints(
        output_dir, plan
    )
    storage_before_table = _artifact_storage(output_dir)
    checkpoint_rows = longrun_common.build_checkpoint_rows(
        plan=plan,
        arrays=arrays,
        backend="histogym",
        preparation_wall_ns=0,
        preparation_cpu_ns=0,
        import_wall_ns=int(import_latency_ns),
        import_cpu_ns=int(import_cpu_ns),
        native_storage=native_storage_by_checkpoint[plan.episode_count],
        harness_storage=storage_before_table["harness_generated"],
    )
    for checkpoint_row in checkpoint_rows:
        checkpoint = int(checkpoint_row["checkpoint_episodes"])
        checkpoint_row["storage"]["native_generated"] = (
            native_storage_by_checkpoint[checkpoint]
        )
    _write_checkpoint_table(checkpoint_csv_path, checkpoint_rows)
    storage_before_result = _artifact_storage(output_dir)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "artifact_type": "histogym_comparison_longrun_result",
        "backend": "histogym",
        "scenario": plan.scenario,
        "status": "complete",
        "started_at_utc": runtime.get("started_at_utc"),
        "ended_at_utc": utc_now(),
        "plan": {
            "path": str(plan.path),
            "file_sha256": plan.file_sha256,
            "content_sha256": plan.content_sha256,
            "protocol_sha256": plan.protocol_sha256,
            "manifest_sha256": plan.manifest_sha256,
            "episode_count": plan.episode_count,
            "checkpoints": list(plan.checkpoints),
        },
        "runtime": runtime,
        "protocol": {
            "accepted_steps_per_episode": 100,
            "tile_size": legacy.TILE_SIZE,
            "native_lifecycle": (
                "reuse one active slide-bound HistoEnv for consecutive same-slide "
                "episodes; close and reconstruct on planned slide change"
            ),
            "render_called": False,
            "diagnosis_actions": 0,
            "stay_actions": 0,
            "policy_or_encoder_loaded": False,
            "latency_scope": "native calls only",
            "checkpoint_writes_during_measurement": False,
            "trace_serialization_after_monitoring": True,
            "checkpoint_native_storage": (
                "post-run exact prefix of empty HistoGym-created constructor "
                "directories; adapter-owned parent final size retained only as "
                "a diagnostic because its historical allocation is unknowable"
            ),
        },
        "aggregate": {
            "episodes": plan.episode_count,
            "accepted_steps": plan.episode_count * 100,
            "attempts": recorder.attempt_count,
            "no_op_attempts": total_no_ops,
            "postponement_events": total_postponements,
            "constructors": constructor_count,
            "expected_constructors_from_plan": expected_constructors,
            "environment_reuses": plan.episode_count - constructor_count,
            "upstream_import_latency_ns": int(import_latency_ns),
            "upstream_import_cpu_ns": int(import_cpu_ns),
            "resource_monitor_samples_at_final_episode_boundary": (
                monitor_samples_at_final_boundary
            ),
            "operation_wall_including_serialization_ns": int(
                time.perf_counter_ns() - operation_started_ns
            ),
        },
        "checkpoints": checkpoint_rows,
        "resources": monitored_resources,
        "trace": trace_metadata,
        "storage": storage_before_result,
        "plot_data": {
            "checkpoint_csv": str(checkpoint_csv_path),
            "checkpoint_json_embedded": True,
        },
    }
    result_path = longrun_common.write_json_exclusive(
        output_dir / "result.json", result
    )
    final_storage = _artifact_storage(output_dir)
    return result_path, result, final_storage


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=pathlib.Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--episode-plan", type=pathlib.Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=("balanced_random_10k", "single_slide_1k"),
        default=None,
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--histogym-root", type=pathlib.Path, default=DEFAULT_UPSTREAM
    )
    parser.add_argument("--monitor-interval", type=float, default=0.10)
    parser.add_argument("--attempt-capacity", type=int, default=102)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.monitor_interval <= 0:
        raise SystemExit("--monitor-interval must be positive")
    if args.attempt_capacity < 100:
        raise SystemExit("--attempt-capacity must be at least 100")
    result_path, result, final_storage = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "backend": result["backend"],
                "scenario": result["scenario"],
                "episodes": result["aggregate"]["episodes"],
                "accepted_steps": result["aggregate"]["accepted_steps"],
                "constructors": result["aggregate"]["constructors"],
                "result": str(result_path),
                "result_sha256": longrun_common.sha256_file(result_path),
                "storage": final_storage,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
