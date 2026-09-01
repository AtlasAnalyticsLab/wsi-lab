#!/usr/bin/env python3
"""Pinned, external benchmark adapter for unmodified HistoGym.

Only the native constructor, reset(), and step() calls are timed. Legacy Gym
return normalization, action selection, validation, and JSON serialization are
outside those call timers. Every native step attempt is retained in the output.
"""

from __future__ import print_function

import argparse
import contextlib
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
import pathlib
import re
import resource
import subprocess
import sys
import time
import traceback
from collections import deque


HERE = pathlib.Path(__file__).resolve().parent
COMPARISON_ROOT = HERE.parent
if str(COMPARISON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARISON_ROOT))

from common.protocol import (  # noqa: E402 - private sibling harness import
    EXPECTED_QUOTAS,
    PAN_PREFERENCE,
    abstract_priorities,
    directory_snapshot,
    latency_summary_ns,
    observation_nbytes,
)
from common.resource_monitor import ResourceMonitor  # noqa: E402


DEFAULT_MANIFEST = HERE.parent / "manifests" / "cohort.json"
DEFAULT_UPSTREAM = HERE / "upstream" / "HistoGym"
EXPECTED_COMMIT = "5b742760321d28a469d2ef9494e886a49d0aa7f7"
TILE_SIZE = 224
ACCEPTED_STEPS_PER_SLIDE = 100
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right", 4: "zoom_in", 5: "zoom_out"}
ACTION_CODES = {name: code for code, name in ACTION_NAMES.items()}
PAN_ROTATION = tuple(ACTION_CODES[name] for name in PAN_PREFERENCE)
SLIDE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PROCESS_PYTHON_START_NS = time.perf_counter_ns()


class ProtocolError(RuntimeError):
    pass


class SlideRunError(RuntimeError):
    def __init__(self, message, record):
        RuntimeError.__init__(self, message)
        self.record = record


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_output(upstream, *args):
    return subprocess.check_output(
        ["git", "-C", str(upstream)] + list(args), text=True
    ).strip()


def verify_upstream(upstream):
    expected_file = upstream / "gymhisto.py"
    if not expected_file.is_file():
        raise ProtocolError("Missing pinned HistoGym checkout: %s" % upstream)
    commit = git_output(upstream, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise ProtocolError(
            "HistoGym commit mismatch: expected %s, found %s"
            % (EXPECTED_COMMIT, commit)
        )
    dirty = git_output(upstream, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ProtocolError("Pinned HistoGym checkout has tracked modifications")
    return commit


def load_histogym(upstream):
    sys.path.insert(0, str(upstream))
    started = time.perf_counter_ns()
    module = importlib.import_module("gymhisto")
    elapsed = time.perf_counter_ns() - started
    actual_file = pathlib.Path(module.__file__).resolve()
    if actual_file != (upstream / "gymhisto.py").resolve():
        raise ProtocolError("Imported gymhisto from the wrong location: %s" % actual_file)
    return module, elapsed


def normalize_reset(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, {}


def normalize_step(result):
    if not isinstance(result, tuple):
        raise ProtocolError("step() did not return a tuple")
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, reward, bool(done), False, info
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), info
    raise ProtocolError("step() returned %d values; expected 4 or 5" % len(result))


def observation_bytes(value):
    return observation_nbytes(value)


def observation_descriptor(observation):
    shape = getattr(observation, "shape", None)
    dtype = getattr(observation, "dtype", None)
    return {
        "bytes": observation_bytes(observation),
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else type(observation).__name__,
    }


def validate_observation(observation):
    descriptor = observation_descriptor(observation)
    expected_shape = [3, TILE_SIZE, TILE_SIZE]
    if descriptor["shape"] != expected_shape:
        raise ProtocolError(
            "Unexpected observation shape %r; expected %r"
            % (descriptor["shape"], expected_shape)
        )
    if descriptor["dtype"] != "uint8":
        raise ProtocolError(
            "Unexpected observation dtype %s; expected uint8" % descriptor["dtype"]
        )
    return descriptor


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return repr(value)


def position(env):
    raw = env.agent_pos
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ProtocolError("Unsupported HistoGym agent_pos: %r" % (raw,))
    return (int(raw[0]), float(raw[1]), float(raw[2]))


def positions_differ(before, after, tolerance=1e-9):
    return before[0] != after[0] or any(
        abs(left - right) > tolerance
        for left, right in zip(before[1:], after[1:])
    )


def relative_magnification(depth, reset_depth):
    return float(2 ** (int(depth) - int(reset_depth)))


def available_pan_actions(env, pan_index):
    depth, x_pos, y_pos = position(env)
    max_x, max_y = env.bound[depth]
    tolerance = 1e-9
    valid = {
        3: x_pos < float(max_x) - tolerance,
        1: y_pos < float(max_y) - tolerance,
        2: x_pos > tolerance,
        0: y_pos > tolerance,
    }
    preferred_offset = pan_index % len(PAN_ROTATION)
    ordered = PAN_ROTATION[preferred_offset:] + PAN_ROTATION[:preferred_offset]
    return [action for action in ordered if valid[action]]


def requested_class_viability(env, requested_class, reset_depth, pan_index):
    """Return native action candidates without mutating the environment."""
    depth = position(env)[0]
    if requested_class == "pan":
        actions = available_pan_actions(env, pan_index)
        return actions, "no_state_changing_pan" if not actions else None
    if requested_class == "zoom_in":
        max_depth = len(env.dz.level_tiles) - 1
        if depth >= max_depth:
            return [], "at_maximum_deepzoom_level"
        return [ACTION_CODES["zoom_in"]], None
    if requested_class == "zoom_out":
        if depth <= reset_depth:
            return [], "at_reset_deepzoom_level"
        return [ACTION_CODES["zoom_out"]], None
    raise ProtocolError("Unknown requested class: %s" % requested_class)


def timed_call(function, quiet_stdout, *args):
    with contextlib.redirect_stdout(quiet_stdout):
        cpu_started = time.process_time_ns()
        wall_started = time.perf_counter_ns()
        result = function(*args)
        wall_elapsed = time.perf_counter_ns() - wall_started
        cpu_elapsed = time.process_time_ns() - cpu_started
    return result, wall_elapsed, cpu_elapsed


def process_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io_values = {}
    try:
        with open("/proc/self/io", "rt") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                io_values[key] = int(value.strip())
    except OSError:
        pass
    current_rss = None
    try:
        with open("/proc/self/statm", "rt") as handle:
            resident_pages = int(handle.read().split()[1])
        current_rss = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        pass
    return {
        "wall_ns": time.perf_counter_ns(),
        "user_ns": int(usage.ru_utime * 1e9),
        "system_ns": int(usage.ru_stime * 1e9),
        "peak_rss_bytes": int(usage.ru_maxrss * 1024),
        "current_rss_bytes": current_rss,
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "io": io_values,
    }


def snapshot_delta(before, after):
    wall_ns = after["wall_ns"] - before["wall_ns"]
    user_ns = after["user_ns"] - before["user_ns"]
    system_ns = after["system_ns"] - before["system_ns"]
    cpu_ns = user_ns + system_ns
    result = {
        "wall_ns": wall_ns,
        "user_cpu_ns": user_ns,
        "system_cpu_ns": system_ns,
        "cpu_ns": cpu_ns,
        "cpu_utilization_percent": (100.0 * cpu_ns / wall_ns) if wall_ns else None,
        "peak_rss_bytes_at_end": after["peak_rss_bytes"],
        "current_rss_bytes_at_start": before["current_rss_bytes"],
        "current_rss_bytes_at_end": after["current_rss_bytes"],
        "minor_page_faults": after["minor_page_faults"] - before["minor_page_faults"],
        "major_page_faults": after["major_page_faults"] - before["major_page_faults"],
        "voluntary_context_switches": (
            after["voluntary_context_switches"] - before["voluntary_context_switches"]
        ),
        "involuntary_context_switches": (
            after["involuntary_context_switches"] - before["involuntary_context_switches"]
        ),
    }
    for key in ("rchar", "wchar", "read_bytes", "write_bytes"):
        if key in before["io"] and key in after["io"]:
            result[key] = after["io"][key] - before["io"][key]
    return result


def latency_summary(values):
    return latency_summary_ns(list(values))


def directory_usage(path):
    return directory_snapshot(path)


def resolve_slide_row(row, cohort):
    if not isinstance(row, dict):
        raise ProtocolError("Manifest %s entry is not an object" % cohort)
    resolved = dict(row)
    if "wsi_path" not in resolved and "image_path" in resolved:
        resolved["wsi_path"] = resolved["image_path"]
    if "xml_path" not in resolved and "annotation_path" in resolved:
        resolved["xml_path"] = resolved["annotation_path"]
    required = ("ordinal", "slide_id", "wsi_path", "xml_path", "label")
    missing = [key for key in required if key not in resolved]
    if missing:
        raise ProtocolError("Manifest entry is missing: %s" % ", ".join(missing))
    if not isinstance(resolved["ordinal"], int):
        raise ProtocolError("ordinal must be an integer for %s" % resolved["slide_id"])
    if not SLIDE_ID_RE.match(str(resolved["slide_id"])):
        raise ProtocolError("Unsafe slide_id: %r" % resolved["slide_id"])
    if resolved["label"] != 1:
        raise ProtocolError("Expected tumor label=1 for %s" % resolved["slide_id"])
    for key in ("wsi_path", "xml_path"):
        path = pathlib.Path(resolved[key]).expanduser().resolve()
        if not path.is_file():
            raise ProtocolError("Missing %s for %s: %s" % (key, resolved["slide_id"], path))
        resolved[key] = str(path)
    return resolved


def load_manifest(path, cohort, limit):
    with path.open("rt") as handle:
        document = json.load(handle)
    key = "slides" if cohort == "measured" else "pilot_slides"
    raw_rows = document.get(key)
    if not isinstance(raw_rows, list):
        raise ProtocolError("Manifest key %r must be a list" % key)
    rows = [resolve_slide_row(row, cohort) for row in raw_rows]
    rows.sort(key=lambda item: item["ordinal"])
    ids = [row["slide_id"] for row in rows]
    ordinals = [row["ordinal"] for row in rows]
    if len(ids) != len(set(ids)) or len(ordinals) != len(set(ordinals)):
        raise ProtocolError("Manifest contains duplicate slide_id or ordinal values")
    if cohort == "measured":
        if len(rows) != 100:
            raise ProtocolError("Measured cohort must contain exactly 100 slides")
        if ordinals != list(range(100)):
            raise ProtocolError("Measured cohort ordinals must be exactly 0..99")
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ProtocolError("Selected cohort is empty")
    return document, rows


def attempt_record(
    accepted_index,
    priority_index,
    postponement_count,
    attempt_index,
    requested_class,
    action,
    before,
    after,
    accepted,
    latency_ns,
    cpu_ns,
    observation,
    reward,
    terminated,
    truncated,
    info,
    reset_depth,
    native_max_depth,
):
    descriptor = validate_observation(observation)
    return {
        "index": accepted_index if accepted else None,
        "priority_index": priority_index,
        "postponement_count": postponement_count,
        "attempt_index": attempt_index,
        "requested_class": requested_class,
        "concrete_action": ACTION_NAMES[action],
        "concrete_action_code": action,
        "accepted": accepted,
        "latency_ns": latency_ns,
        "cpu_ns": cpu_ns,
        "depth_before": before[0],
        "depth_after": after[0],
        "at_native_max_depth_before": before[0] == native_max_depth,
        "at_native_max_depth_after": after[0] == native_max_depth,
        "position_before": list(before),
        "position_after": list(after),
        "relative_magnification_before": relative_magnification(before[0], reset_depth),
        "relative_magnification_after": relative_magnification(after[0], reset_depth),
        "observation_bytes": descriptor["bytes"],
        "observation_shape": descriptor["shape"],
        "observation_dtype": descriptor["dtype"],
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "info": json_safe(info),
    }


def run_native_step(
    env,
    action,
    accepted_index,
    priority_index,
    postponement_count,
    attempt_index,
    requested_class,
    reset_depth,
    native_max_depth,
    quiet_stdout,
):
    before = position(env)
    raw_result, latency_ns, cpu_ns = timed_call(env.step, quiet_stdout, action)
    after = position(env)
    observation, reward, terminated, truncated, info = normalize_step(raw_result)
    accepted = positions_differ(before, after)
    record = attempt_record(
        accepted_index,
        priority_index,
        postponement_count,
        attempt_index,
        requested_class,
        action,
        before,
        after,
        accepted,
        latency_ns,
        cpu_ns,
        observation,
        reward,
        terminated,
        truncated,
        info,
        reset_depth,
        native_max_depth,
    )
    return record


def run_slide(histogym, slide, output_dir, quiet_stdout):
    native_result_path = output_dir / "native_results" / (
        "%03d_%s" % (slide["ordinal"], slide["slide_id"])
    )
    record = {
        "ordinal": slide["ordinal"],
        "slide_id": slide["slide_id"],
        "wsi_path": slide["wsi_path"],
        "xml_path": slide["xml_path"],
        "label": slide["label"],
        "status": "running",
        "priority_order": list(abstract_priorities(slide["ordinal"])),
        "postponements": [],
        "attempts": [],
        "steps": [],
    }
    env = None
    slide_resource_start = process_snapshot()
    lifecycle_outer_started_ns = time.perf_counter_ns()
    online_outer_started_ns = None
    try:
        constructor_args = (
            slide["wsi_path"],
            slide["xml_path"],
            TILE_SIZE,
            str(native_result_path),
        )
        env, constructor_ns, constructor_cpu_ns = timed_call(
            histogym.HistoEnv, quiet_stdout, *constructor_args
        )
        record["construction"] = {
            "latency_ns": constructor_ns,
            "cpu_ns": constructor_cpu_ns,
            "native_result_path": str(native_result_path),
        }

        online_outer_started_ns = time.perf_counter_ns()
        raw_reset, reset_ns, reset_cpu_ns = timed_call(env.reset, quiet_stdout)
        reset_observation, reset_info = normalize_reset(raw_reset)
        reset_descriptor = validate_observation(reset_observation)
        reset_depth = position(env)[0]
        native_max_depth = len(env.dz.level_tiles) - 1
        record["reset"] = {
            "latency_ns": reset_ns,
            "cpu_ns": reset_cpu_ns,
            "position": list(position(env)),
            "depth": reset_depth,
            "native_max_depth": native_max_depth,
            "available_zoom_in_levels": native_max_depth - reset_depth,
            "relative_magnification": 1.0,
            "observation_bytes": reset_descriptor["bytes"],
            "observation_shape": reset_descriptor["shape"],
            "observation_dtype": reset_descriptor["dtype"],
            "info": json_safe(reset_info),
        }

        pending = deque(
            {
                "priority_index": priority_index,
                "requested_class": requested_class,
                "postponement_count": 0,
                "unexpected_no_op_count": 0,
            }
            for priority_index, requested_class in enumerate(record["priority_order"])
        )
        pan_index = 0
        attempt_index = 0
        while pending:
            selected = None
            selected_actions = None
            selected_selection_ns = 0
            scan_length = len(pending)
            for _ in range(scan_length):
                item = pending.popleft()
                selection_started = time.perf_counter_ns()
                actions, unavailable_reason = requested_class_viability(
                    env, item["requested_class"], reset_depth, pan_index
                )
                selection_ns = time.perf_counter_ns() - selection_started
                if actions:
                    selected = item
                    selected_actions = actions
                    selected_selection_ns = selection_ns
                    break
                item["postponement_count"] += 1
                record["postponements"].append(
                    {
                        "event_index": len(record["postponements"]),
                        "accepted_actions_so_far": len(record["steps"]),
                        "priority_index": item["priority_index"],
                        "requested_class": item["requested_class"],
                        "postponement_count": item["postponement_count"],
                        "reason": unavailable_reason,
                        "position": list(position(env)),
                        "selection_latency_ns": selection_ns,
                    }
                )
                pending.append(item)
            if selected is None:
                raise ProtocolError(
                    "No pending abstract class can change state at position %r"
                    % (position(env),)
                )

            accepted_record = None
            for action in selected_actions:
                attempt = run_native_step(
                    env,
                    action,
                    len(record["steps"]),
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
                record["attempts"].append(attempt)
                attempt_index += 1
                if attempt["terminated"] or attempt["truncated"]:
                    raise ProtocolError(
                        "HistoGym ended before 100 accepted actions at step %d"
                        % len(record["steps"])
                    )
                if attempt["accepted"]:
                    accepted_record = dict(attempt)
                    break

            if accepted_record is None:
                selected["unexpected_no_op_count"] += 1
                selected["postponement_count"] += 1
                record["postponements"].append(
                    {
                        "event_index": len(record["postponements"]),
                        "accepted_actions_so_far": len(record["steps"]),
                        "priority_index": selected["priority_index"],
                        "requested_class": selected["requested_class"],
                        "postponement_count": selected["postponement_count"],
                        "reason": "all_predicted_native_actions_were_no_ops",
                        "position": list(position(env)),
                        "selection_latency_ns": 0,
                    }
                )
                if selected["unexpected_no_op_count"] > 2:
                    raise ProtocolError(
                        "Repeated unexpected no-op for priority item %d"
                        % selected["priority_index"]
                    )
                pending.append(selected)
                continue

            record["steps"].append(accepted_record)
            if selected["requested_class"] == "pan":
                pan_index += 1

        record["outer_wall"] = {
            "reset_through_rollout_ns": time.perf_counter_ns() - online_outer_started_ns,
            "construction_through_rollout_ns": (
                time.perf_counter_ns() - lifecycle_outer_started_ns
            ),
            "includes_adapter_and_postponement_overhead": True,
            "excludes_cleanup": True,
        }

        if len(record["steps"]) != ACCEPTED_STEPS_PER_SLIDE:
            raise ProtocolError("Episode did not contain exactly 100 accepted actions")
        counts = {
            action_class: sum(
                1 for step in record["steps"] if step["requested_class"] == action_class
            )
            for action_class in EXPECTED_QUOTAS
        }
        if counts != EXPECTED_QUOTAS:
            raise ProtocolError("Accepted action quotas are wrong: %r" % counts)

        record["summary"] = summarize_slide(record)
        record["status"] = "complete"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise SlideRunError(str(error), record)
    finally:
        if "outer_wall" not in record:
            record["outer_wall"] = {
                "reset_through_rollout_ns": (
                    time.perf_counter_ns() - online_outer_started_ns
                    if online_outer_started_ns is not None
                    else None
                ),
                "construction_through_rollout_ns": (
                    time.perf_counter_ns() - lifecycle_outer_started_ns
                ),
                "includes_adapter_and_postponement_overhead": True,
                "excludes_cleanup": True,
                "partial": True,
            }
        if env is not None:
            try:
                _, close_ns, close_cpu_ns = timed_call(env.close, quiet_stdout)
                record["native_close"] = {
                    "latency_ns": close_ns,
                    "cpu_ns": close_cpu_ns,
                }
            except BaseException as close_error:
                record["native_close_error"] = repr(close_error)
            adapter_close_started = time.perf_counter_ns()
            try:
                slide_handle = getattr(env, "slide", None)
                if slide_handle is not None and hasattr(slide_handle, "close"):
                    slide_handle.close()
            finally:
                record["adapter_slide_close_latency_ns"] = (
                    time.perf_counter_ns() - adapter_close_started
                )
        slide_resource_end = process_snapshot()
        record["resources"] = snapshot_delta(slide_resource_start, slide_resource_end)
    return record


def summarize_slide(record):
    attempts = record["attempts"]
    accepted = record["steps"]
    no_ops = [attempt for attempt in attempts if not attempt["accepted"]]
    attempt_latency = [attempt["latency_ns"] for attempt in attempts]
    accepted_latency = [step["latency_ns"] for step in accepted]
    native_online_ns = record["reset"]["latency_ns"] + sum(attempt_latency)
    depths = [record["reset"]["depth"]] + [step["depth_after"] for step in accepted]
    deepest_depth = max(depths)
    depth_occupancy = {}
    relative_depth_occupancy = {}
    for step in accepted:
        depth_key = str(step["depth_after"])
        relative_key = str(step["depth_after"] - record["reset"]["depth"])
        depth_occupancy[depth_key] = depth_occupancy.get(depth_key, 0) + 1
        relative_depth_occupancy[relative_key] = (
            relative_depth_occupancy.get(relative_key, 0) + 1
        )
    return {
        "accepted_actions": len(accepted),
        "native_step_calls": len(attempts),
        "no_op_calls": len(no_ops),
        "no_op_rate": float(len(no_ops)) / len(attempts) if attempts else 0.0,
        "accepted_action_counts": {
            key: sum(1 for step in accepted if step["requested_class"] == key)
            for key in EXPECTED_QUOTAS
        },
        "construction_latency_ns": record["construction"]["latency_ns"],
        "reset_latency_ns": record["reset"]["latency_ns"],
        "accepted_step_latency": latency_summary(accepted_latency),
        "all_native_step_call_latency": latency_summary(attempt_latency),
        "native_online_latency_ns": native_online_ns,
        "outer_reset_through_rollout_ns": record["outer_wall"][
            "reset_through_rollout_ns"
        ],
        "outer_construction_through_rollout_ns": record["outer_wall"][
            "construction_through_rollout_ns"
        ],
        "accepted_steps_per_native_step_second": (
            1e9 * len(accepted) / sum(attempt_latency) if attempt_latency else None
        ),
        "accepted_steps_per_online_second_including_reset": (
            1e9 * len(accepted) / native_online_ns if native_online_ns else None
        ),
        "reset_observation_bytes": record["reset"]["observation_bytes"],
        "step_observation_bytes": sorted(
            set(step["observation_bytes"] for step in accepted)
        ),
        "minimum_depth_reached": min(depths),
        "maximum_depth_reached": deepest_depth,
        "native_max_depth": record["reset"]["native_max_depth"],
        "reached_native_max_depth": deepest_depth == record["reset"]["native_max_depth"],
        "accepted_steps_at_native_max_depth": sum(
            1 for step in accepted if step["at_native_max_depth_after"]
        ),
        "depth_occupancy_after": depth_occupancy,
        "relative_depth_occupancy_after": relative_depth_occupancy,
    }


def aggregate_results(slides):
    completed = [slide for slide in slides if slide.get("status") == "complete"]
    attempts = [attempt for slide in completed for attempt in slide["attempts"]]
    accepted = [step for slide in completed for step in slide["steps"]]
    no_ops = [attempt for attempt in attempts if not attempt["accepted"]]
    construction = [slide["construction"]["latency_ns"] for slide in completed]
    resets = [slide["reset"]["latency_ns"] for slide in completed]
    accepted_latencies = [step["latency_ns"] for step in accepted]
    attempt_latencies = [attempt["latency_ns"] for attempt in attempts]
    native_online_ns = sum(resets) + sum(attempt_latencies)
    outer_online = [
        slide["outer_wall"]["reset_through_rollout_ns"] for slide in completed
    ]
    outer_lifecycle = [
        slide["outer_wall"]["construction_through_rollout_ns"]
        for slide in completed
    ]
    reached_native_maximum = sum(
        1 for slide in completed if slide["summary"]["reached_native_max_depth"]
    )
    aggregate_relative_depth_occupancy = {}
    for slide in completed:
        for depth, count in slide["summary"]["relative_depth_occupancy_after"].items():
            aggregate_relative_depth_occupancy[depth] = (
                aggregate_relative_depth_occupancy.get(depth, 0) + count
            )
    by_class = {}
    for action_class in EXPECTED_QUOTAS:
        values = [
            step["latency_ns"]
            for step in accepted
            if step["requested_class"] == action_class
        ]
        by_class[action_class] = latency_summary(values)
    return {
        "selected_slides": len(slides),
        "completed_slides": len(completed),
        "accepted_actions": len(accepted),
        "native_step_calls": len(attempts),
        "no_op_calls": len(no_ops),
        "no_op_rate": float(len(no_ops)) / len(attempts) if attempts else 0.0,
        "construction_latency": latency_summary(construction),
        "reset_latency": latency_summary(resets),
        "accepted_step_latency": latency_summary(accepted_latencies),
        "all_native_step_call_latency": latency_summary(attempt_latencies),
        "accepted_step_latency_by_class": by_class,
        "total_construction_ns": sum(construction),
        "total_native_online_ns": native_online_ns,
        "outer_reset_through_rollout": latency_summary(outer_online),
        "outer_construction_through_rollout": latency_summary(outer_lifecycle),
        "total_outer_reset_through_rollout_ns": sum(outer_online),
        "total_outer_construction_through_rollout_ns": sum(outer_lifecycle),
        "accepted_steps_per_native_step_second": (
            1e9 * len(accepted) / sum(attempt_latencies) if attempt_latencies else None
        ),
        "accepted_steps_per_online_second_including_resets": (
            1e9 * len(accepted) / native_online_ns if native_online_ns else None
        ),
        "observation_bytes_per_reset": sorted(
            set(slide["reset"]["observation_bytes"] for slide in completed)
        ),
        "observation_bytes_per_step": sorted(
            set(step["observation_bytes"] for step in accepted)
        ),
        "slides_reaching_native_max_depth": reached_native_maximum,
        "fraction_slides_reaching_native_max_depth": (
            float(reached_native_maximum) / len(completed) if completed else None
        ),
        "accepted_steps_at_native_max_depth": sum(
            slide["summary"]["accepted_steps_at_native_max_depth"]
            for slide in completed
        ),
        "relative_depth_occupancy_after": aggregate_relative_depth_occupancy,
    }


def dependency_versions():
    names = (
        "gym",
        "numpy",
        "Pillow",
        "Shapely",
        "matplotlib",
        "opencv-python",
        "openslide-python",
        "psutil",
    )
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    try:
        import openslide

        result["openslide_library"] = getattr(openslide, "__library_version__", None)
    except ImportError:
        result["openslide_library"] = None
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort", choices=("measured", "pilot"), default="measured")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--histogym-root", type=pathlib.Path, default=DEFAULT_UPSTREAM)
    return parser


def write_result(output_dir, result):
    destination = output_dir / "result.json"
    temporary = output_dir / "result.json.tmp"
    with temporary.open("wt") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(destination))
    return destination


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.cohort == "measured" and args.limit is not None:
        raise SystemExit("--limit is forbidden for the 100-slide measured cohort")
    if args.repetition < 0:
        raise SystemExit("--repetition must be non-negative")
    if args.cohort == "measured" and args.repetition not in (1, 2, 3):
        raise SystemExit("Measured HistoGym repetitions must be exactly 1, 2, or 3")

    manifest_path = args.manifest.expanduser().resolve()
    upstream = args.histogym_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == upstream or upstream in output_dir.parents:
        raise SystemExit("Output directory must not be inside the pinned checkout")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("Output directory is not empty: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    commit = verify_upstream(upstream)
    _, slides = load_manifest(manifest_path, args.cohort, args.limit)
    input_bytes = sum(
        pathlib.Path(slide[key]).stat().st_size
        for slide in slides
        for key in ("wsi_path", "xml_path")
    )

    import_cpu_started_ns = time.process_time_ns()
    histogym, import_latency_ns = load_histogym(upstream)
    import_cpu_ns = time.process_time_ns() - import_cpu_started_ns
    # Import is preparation and has its own wall/CPU fields. Start the online
    # process-resource baseline only after all native libraries are resident,
    # matching the WSI-Lab online resource scope.
    resource_monitor = ResourceMonitor(interval_seconds=0.10, gpu_index=None).start()
    started_at = utc_now()
    result = {
        "schema_version": 1,
        "backend": "histogym",
        "status": "running",
        "repetition": args.repetition,
        "cohort": args.cohort,
        "started_at": started_at,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "selected_rows": len(slides),
        },
        "upstream": {
            "repository": "https://github.com/xjtuai/histogym.git",
            "commit": commit,
            "path": str(upstream),
            "tracked_worktree_clean": True,
        },
        "runtime": {
            "python": sys.version,
            "dependencies": dependency_versions(),
            "upstream_import_latency_ns": import_latency_ns,
            "upstream_import_cpu_ns": import_cpu_ns,
            "python_code_start_to_import_complete_ns": (
                time.perf_counter_ns() - PROCESS_PYTHON_START_NS
            ),
        },
        "protocol": {
            "tile_size": TILE_SIZE,
            "accepted_actions_per_slide": ACCEPTED_STEPS_PER_SLIDE,
            "abstract_priority_seed_rule": "1 + slide ordinal",
            "unavailable_class_handling": "queue and postpone until state-changing",
            "accepted_action_quotas": EXPECTED_QUOTAS,
            "pan_preference_rotation": [ACTION_NAMES[action] for action in PAN_ROTATION],
            "render_called": False,
            "diagnosis_actions": False,
            "native_stdout_suppressed": True,
            "latency_scope": (
                "native calls only; action selection, legacy API normalization, "
                "validation, and serialization excluded"
            ),
            "resource_scope": (
                "native per-slide construction, reset, steps, and close after "
                "HistoGym/native-library import; final serialization excluded"
            ),
            "percentile_method": "shared linear interpolation",
        },
        "input_storage": {
            "selected_wsi_and_xml_logical_bytes": input_bytes,
            "preexisting_and_not_generated_by_benchmark": True,
        },
        "slides": [],
    }

    failed = None
    with open(os.devnull, "w") as quiet_stdout:
        for position_index, slide in enumerate(slides, start=1):
            print(
                "HistoGym [%d/%d] %s" % (position_index, len(slides), slide["slide_id"]),
                file=sys.stderr,
                flush=True,
            )
            try:
                slide_result = run_slide(histogym, slide, output_dir, quiet_stdout)
                result["slides"].append(slide_result)
            except SlideRunError as error:
                result["slides"].append(error.record)
                failed = error
                break

    monitored_resources = resource_monitor.stop()
    result["ended_at"] = utc_now()
    result["aggregate"] = aggregate_results(result["slides"])
    result["resources"] = monitored_resources
    result["native_generated_storage"] = directory_usage(
        output_dir / "native_results"
    )
    if failed is None and len(result["slides"]) == len(slides):
        result["status"] = "complete"
    else:
        result["status"] = "failed"
        result["failure"] = {
            "type": type(failed).__name__ if failed is not None else "IncompleteRun",
            "message": str(failed) if failed is not None else "Not all slides ran",
        }
    destination = write_result(output_dir, result)
    print("Wrote %s" % destination, file=sys.stderr)
    if result["status"] != "complete":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
