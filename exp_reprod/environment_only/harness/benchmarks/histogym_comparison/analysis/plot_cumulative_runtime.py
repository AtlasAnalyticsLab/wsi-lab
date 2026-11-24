#!/usr/bin/env python3
"""Create a dense cumulative-runtime plot from immutable long-run traces.

This is post-processing only.  It reads the saved per-episode cumulative wall
clock counters and never imports, constructs, resets, or steps either backend.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

HARNESS_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "balanced_random_10k"
EXPECTED_EPISODES = 10_000
EXPECTED_ACCEPTED_STEPS_PER_EPISODE = 100
TRACE_ARRAY = "episode_cumulative_online_wall_ns"
DISPLAY_NAMES = {"wsi_lab": "WSI-Lab", "histogym": "HistoGym"}
RESULT_BACKEND_NAMES = {"wsi_lab": "wsi-lab", "histogym": "histogym"}


class CumulativeRuntimeError(RuntimeError):
    """Raised when an input or derived cumulative-runtime artifact is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CumulativeRuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CumulativeRuntimeError(f"cannot read {label}: {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


@dataclass(frozen=True)
class BackendCurve:
    backend: str
    result_path: Path
    result_sha256: str
    trace_path: Path
    trace_sha256: str
    plan: dict[str, Any]
    preparation_ns: int
    import_ns: int
    cumulative_online_ns: np.ndarray
    checkpoints: tuple[dict[str, Any], ...]

    @property
    def initial_ns(self) -> int:
        return self.preparation_ns + self.import_ns

    @property
    def full_cumulative_ns(self) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray([self.initial_ns], dtype=np.uint64),
                self.cumulative_online_ns + np.uint64(self.initial_ns),
            )
        )


def _resolved_file(value: Path, label: str) -> Path:
    path = value.expanduser().resolve()
    require(path.is_file(), f"missing {label}: {path}")
    return path


def _as_nonnegative_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} is not an integer")
    require(value >= 0, f"{label} is negative")
    return value


def load_backend_curve(
    result_path: Path,
    backend: str,
    *,
    expected_episodes: int = EXPECTED_EPISODES,
    expected_steps_per_episode: int = EXPECTED_ACCEPTED_STEPS_PER_EPISODE,
) -> BackendCurve:
    """Load and fully validate one backend's dense cumulative wall-clock trace."""

    require(backend in DISPLAY_NAMES, f"unknown backend: {backend}")
    result_path = _resolved_file(result_path, f"{backend} result")
    result_sha256 = sha256_file(result_path)
    result = load_json(result_path, f"{backend} result")
    require(sha256_file(result_path) == result_sha256, f"{backend} result changed while read")
    require(result.get("status") == "complete", f"{backend} result is incomplete")
    require(result.get("scenario") == SCENARIO, f"{backend} result has the wrong scenario")
    require(
        result.get("backend") == RESULT_BACKEND_NAMES[backend],
        f"{backend} result has the wrong backend identity",
    )

    plan_key = "episode_plan" if backend == "wsi_lab" else "plan"
    plan_value = result.get(plan_key)
    require(isinstance(plan_value, dict), f"{backend} result has no bound episode plan")
    plan = dict(cast(dict[str, Any], plan_value))
    require(plan.get("episode_count") == expected_episodes, f"{backend} episode count drifted")
    if backend == "wsi_lab":
        require(
            plan.get("accepted_steps_per_episode") == expected_steps_per_episode,
            "WSI-Lab accepted-step quota drifted",
        )
    for name in ("content_sha256", "file_sha256", "manifest_sha256", "protocol_sha256"):
        value = plan.get(name)
        require(isinstance(value, str) and len(value) == 64, f"{backend} plan {name} is invalid")

    trace_metadata_value = result.get("trace")
    require(isinstance(trace_metadata_value, dict), f"{backend} trace metadata is missing")
    trace_metadata = cast(dict[str, Any], trace_metadata_value)
    require(trace_metadata.get("backend") == backend, f"{backend} trace identity differs")
    require(trace_metadata.get("scenario") == SCENARIO, f"{backend} trace scenario differs")
    schema_value = trace_metadata.get("array_schema")
    descriptor_value = trace_metadata.get("trace")
    require(isinstance(schema_value, dict), f"{backend} trace schema is missing")
    require(isinstance(descriptor_value, dict), f"{backend} trace descriptor is missing")
    schema = cast(dict[str, Any], schema_value)
    descriptor = cast(dict[str, Any], descriptor_value)
    array_schema_value = schema.get(TRACE_ARRAY)
    require(isinstance(array_schema_value, dict), f"{backend} trace lacks {TRACE_ARRAY}")
    array_schema = cast(dict[str, Any], array_schema_value)
    require(
        array_schema.get("shape") == [expected_episodes],
        f"{backend} cumulative trace shape differs",
    )
    require(
        descriptor.get("episode_count") == expected_episodes,
        f"{backend} trace episode count differs",
    )
    require(
        descriptor.get("accepted_count") == expected_episodes * expected_steps_per_episode,
        f"{backend} trace accepted-step count differs",
    )
    trace_path_value = descriptor.get("path")
    expected_trace_sha256 = descriptor.get("sha256")
    require(isinstance(trace_path_value, str), f"{backend} trace path is invalid")
    require(
        isinstance(expected_trace_sha256, str) and len(expected_trace_sha256) == 64,
        f"{backend} trace hash is invalid",
    )
    trace_path = _resolved_file(Path(cast(str, trace_path_value)), f"{backend} compact trace")
    trace_sha256 = sha256_file(trace_path)
    require(trace_sha256 == expected_trace_sha256, f"{backend} compact trace hash differs")
    with np.load(trace_path, allow_pickle=False) as arrays:
        require(TRACE_ARRAY in arrays.files, f"{backend} compact trace lacks {TRACE_ARRAY}")
        cumulative = np.asarray(arrays[TRACE_ARRAY]).copy()
    require(sha256_file(trace_path) == trace_sha256, f"{backend} trace changed while read")
    require(cumulative.shape == (expected_episodes,), f"{backend} cumulative array shape differs")
    require(cumulative.dtype == np.dtype(np.uint64), f"{backend} cumulative array dtype differs")
    require(
        np.dtype(str(array_schema.get("dtype"))) == cumulative.dtype,
        f"{backend} cumulative array disagrees with its schema",
    )
    require(bool(np.all(cumulative[1:] > cumulative[:-1])), f"{backend} trace is not increasing")
    require(int(cumulative[0]) > 0, f"{backend} first episode has zero online wall time")

    checkpoints_value = result.get("checkpoints")
    require(
        bool(isinstance(checkpoints_value, list) and checkpoints_value),
        f"{backend} checkpoints missing",
    )
    checkpoints = tuple(cast(list[dict[str, Any]], checkpoints_value))
    checkpoint_numbers: list[int] = []
    preparation_ns: int | None = None
    import_ns: int | None = None
    for checkpoint in checkpoints:
        require(isinstance(checkpoint, dict), f"{backend} checkpoint is not an object")
        episode = _as_nonnegative_int(
            checkpoint.get("checkpoint_episodes"), f"{backend} checkpoint episode"
        )
        require(1 <= episode <= expected_episodes, f"{backend} checkpoint is out of range")
        checkpoint_numbers.append(episode)
        require(
            checkpoint.get("accepted_steps") == episode * expected_steps_per_episode,
            f"{backend} checkpoint {episode} accepted-step count differs",
        )
        require(
            checkpoint.get("plan_content_sha256") == plan["content_sha256"],
            f"{backend} checkpoint {episode} plan binding differs",
        )
        components_value = checkpoint.get("components_ns")
        require(
            isinstance(components_value, dict),
            f"{backend} checkpoint {episode} components missing",
        )
        components = cast(dict[str, Any], components_value)
        checkpoint_preparation = _as_nonnegative_int(
            components.get("preparation_wall"), f"{backend} preprocessing time"
        )
        checkpoint_import = _as_nonnegative_int(
            components.get("import_wall"), f"{backend} import time"
        )
        if preparation_ns is None:
            preparation_ns = checkpoint_preparation
            import_ns = checkpoint_import
        require(
            checkpoint_preparation == preparation_ns and checkpoint_import == import_ns,
            f"{backend} fixed preparation/import time changed across checkpoints",
        )
        online_ns = int(cumulative[episode - 1])
        expected_full_ns = checkpoint_preparation + checkpoint_import + online_ns
        require(
            components.get("cumulative_online_outer_wall") == online_ns,
            f"{backend} checkpoint {episode} disagrees with the dense online trace",
        )
        require(
            components.get("full_preparation_import_online_wall") == expected_full_ns,
            f"{backend} checkpoint {episode} full cumulative formula differs",
        )
    require(checkpoint_numbers == sorted(set(checkpoint_numbers)), f"{backend} checkpoints differ")
    require(checkpoint_numbers[-1] == expected_episodes, f"{backend} final checkpoint is missing")
    require(preparation_ns is not None and import_ns is not None, f"{backend} timing base missing")

    return BackendCurve(
        backend=backend,
        result_path=result_path,
        result_sha256=result_sha256,
        trace_path=trace_path,
        trace_sha256=trace_sha256,
        plan=plan,
        preparation_ns=cast(int, preparation_ns),
        import_ns=cast(int, import_ns),
        cumulative_online_ns=cumulative,
        checkpoints=checkpoints,
    )


def validate_pair(wsi: BackendCurve, histogym: BackendCurve) -> None:
    require(wsi.backend == "wsi_lab", "first curve is not WSI-Lab")
    require(histogym.backend == "histogym", "second curve is not HistoGym")
    for field in ("content_sha256", "file_sha256", "manifest_sha256", "protocol_sha256"):
        require(wsi.plan[field] == histogym.plan[field], f"cross-backend plan {field} differs")
    require(
        wsi.cumulative_online_ns.shape == histogym.cumulative_online_ns.shape,
        "cross-backend episode counts differ",
    )


def first_crossover_episode(wsi: BackendCurve, histogym: BackendCurve) -> int:
    """Return the first episode where HistoGym cumulative time reaches WSI-Lab."""

    wsi_full = wsi.full_cumulative_ns
    histogym_full = histogym.full_cumulative_ns
    require(histogym_full[0] < wsi_full[0], "HistoGym does not start below WSI-Lab")
    crossing = np.flatnonzero(histogym_full[1:] >= wsi_full[1:])
    require(bool(crossing.size), "the cumulative curves never cross")
    return int(crossing[0]) + 1


def _write_csv(path: Path, wsi: BackendCurve, histogym: BackendCurve) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    fieldnames = [
        "episode",
        "wsi_lab_preprocessing_ns",
        "wsi_lab_import_ns",
        "wsi_lab_cumulative_online_wall_ns",
        "wsi_lab_full_cumulative_wall_ns",
        "wsi_lab_full_cumulative_seconds",
        "wsi_lab_full_cumulative_minutes",
        "histogym_preprocessing_ns",
        "histogym_import_ns",
        "histogym_cumulative_online_wall_ns",
        "histogym_full_cumulative_wall_ns",
        "histogym_full_cumulative_seconds",
        "histogym_full_cumulative_minutes",
    ]
    wsi_online = np.concatenate((np.asarray([0], dtype=np.uint64), wsi.cumulative_online_ns))
    histo_online = np.concatenate((np.asarray([0], dtype=np.uint64), histogym.cumulative_online_ns))
    wsi_full = wsi.full_cumulative_ns
    histo_full = histogym.full_cumulative_ns
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for episode in range(len(wsi_full)):
            writer.writerow(
                {
                    "episode": episode,
                    "wsi_lab_preprocessing_ns": wsi.preparation_ns,
                    "wsi_lab_import_ns": wsi.import_ns,
                    "wsi_lab_cumulative_online_wall_ns": int(wsi_online[episode]),
                    "wsi_lab_full_cumulative_wall_ns": int(wsi_full[episode]),
                    "wsi_lab_full_cumulative_seconds": f"{int(wsi_full[episode]) / 1e9:.9f}",
                    "wsi_lab_full_cumulative_minutes": f"{int(wsi_full[episode]) / 60e9:.9f}",
                    "histogym_preprocessing_ns": histogym.preparation_ns,
                    "histogym_import_ns": histogym.import_ns,
                    "histogym_cumulative_online_wall_ns": int(histo_online[episode]),
                    "histogym_full_cumulative_wall_ns": int(histo_full[episode]),
                    "histogym_full_cumulative_seconds": f"{int(histo_full[episode]) / 1e9:.9f}",
                    "histogym_full_cumulative_minutes": f"{int(histo_full[episode]) / 60e9:.9f}",
                }
            )


def _write_plot(
    png_path: Path,
    pdf_path: Path,
    wsi: BackendCurve,
    histogym: BackendCurve,
    crossover: int,
    *,
    show_wsi_start_annotation: bool = True,
    show_scope_subtitle: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    require(not png_path.exists(), f"refusing to overwrite {png_path}")
    require(not pdf_path.exists(), f"refusing to overwrite {pdf_path}")
    episodes = np.arange(len(wsi.full_cumulative_ns))
    wsi_minutes = wsi.full_cumulative_ns.astype(np.float64) / 60e9
    histo_minutes = histogym.full_cumulative_ns.astype(np.float64) / 60e9

    fig, axis = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    axis.plot(episodes, wsi_minutes, color="#0072B2", linewidth=2.2, label="WSI-Lab")
    axis.plot(episodes, histo_minutes, color="#D55E00", linewidth=2.2, label="HistoGym")
    crossing_minutes = float((wsi_minutes[crossover] + histo_minutes[crossover]) / 2)
    axis.axvline(crossover, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.scatter(
        [crossover, crossover],
        [wsi_minutes[crossover], histo_minutes[crossover]],
        color=["#0072B2", "#D55E00"],
        s=25,
        zorder=4,
    )
    axis.annotate(
        f"Break-even: episode {crossover:,}",
        xy=(crossover, crossing_minutes),
        xytext=(18, 22),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BBBBBB", "alpha": 0.9},
    )
    axis.scatter([0], [wsi_minutes[0]], color="#0072B2", s=25, zorder=4)
    if show_wsi_start_annotation:
        axis.annotate(
            f"WSI-Lab starts at {wsi_minutes[0]:.1f} min",
            xy=(0, wsi_minutes[0]),
            xytext=(20, 8),
            textcoords="offset points",
            fontsize=8.5,
            color="#005A8D",
        )
    axis.annotate(
        f"{histo_minutes[-1]:.1f} min",
        xy=(episodes[-1], histo_minutes[-1]),
        xytext=(-8, 7),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#A84600",
    )
    axis.annotate(
        f"{wsi_minutes[-1]:.1f} min",
        xy=(episodes[-1], wsi_minutes[-1]),
        xytext=(-8, -14),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#005A8D",
    )
    title = "Cumulative runtime over 10,000 episodes"
    if show_scope_subtitle:
        title += "\nFresh preprocessing + import/setup + online interaction"
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Episode number")
    axis.set_ylabel("Cumulative wall time (minutes)")
    axis.set_xlim(0, EXPECTED_EPISODES)
    axis.set_ylim(bottom=0)
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def _point_record(episode: int, wsi: BackendCurve, histogym: BackendCurve) -> dict[str, Any]:
    wsi_ns = int(wsi.full_cumulative_ns[episode])
    histo_ns = int(histogym.full_cumulative_ns[episode])
    return {
        "episode": episode,
        "wsi_lab_full_cumulative_ns": wsi_ns,
        "wsi_lab_full_cumulative_seconds": wsi_ns / 1e9,
        "histogym_full_cumulative_ns": histo_ns,
        "histogym_full_cumulative_seconds": histo_ns / 1e9,
        "histogym_minus_wsi_lab_seconds": (histo_ns - wsi_ns) / 1e9,
    }


def _write_json(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_text(path: Path, value: str) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2: Any = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is required for exclusive publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise CumulativeRuntimeError(f"refusing to replace existing output: {destination}")
    raise CumulativeRuntimeError(
        f"cannot atomically publish {destination}: {os.strerror(error_number)}"
    )


def _protect_output(output: Path, curves: Sequence[BackendCurve]) -> None:
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    for curve in curves:
        for protected in (curve.result_path.parent, curve.trace_path.parent):
            require(
                output != protected
                and output not in protected.parents
                and protected not in output.parents,
                f"derived output overlaps measured input storage: {protected}",
            )


def _readme(wsi: BackendCurve, histogym: BackendCurve, crossover: int) -> str:
    points = {episode: _point_record(episode, wsi, histogym) for episode in (0, 2_000, 10_000)}
    return f"""# Cumulative runtime: balanced-random 10k

The y-axis is total measured wall time through episode N, not time per episode:

`preprocessing + import/setup + cumulative online wall time through episode N`

The episode-0 point is preprocessing plus import/setup. The lines then use every exact
per-episode boundary saved in the compact traces; they are not interpolated from checkpoints.

| Episode | WSI-Lab | HistoGym |
|---:|---:|---:|
| 0 | {points[0]["wsi_lab_full_cumulative_seconds"]:.2f} s | {points[0]["histogym_full_cumulative_seconds"]:.2f} s |
| 2,000 | {points[2000]["wsi_lab_full_cumulative_seconds"]:.2f} s | {points[2000]["histogym_full_cumulative_seconds"]:.2f} s |
| 10,000 | {points[10000]["wsi_lab_full_cumulative_seconds"]:.2f} s | {points[10000]["histogym_full_cumulative_seconds"]:.2f} s |

HistoGym first reaches/exceeds WSI-Lab's cumulative time at episode **{crossover:,}**.

The totals exclude cache-drop waiting, output serialization, post-hoc tissue analysis,
belief/policy inference, online embedding reads, rendering, and original UNI-v2 embedding
generation. This is one measured run per backend, so the figure has no confidence band.
"""


def generate(
    wsi_result: Path,
    histogym_result: Path,
    output: Path,
    *,
    show_wsi_start_annotation: bool = True,
    show_scope_subtitle: bool = True,
) -> Path:
    analyzer_path = Path(__file__).resolve()
    analyzer_sha256 = sha256_file(analyzer_path)
    wsi = load_backend_curve(wsi_result, "wsi_lab")
    histogym = load_backend_curve(histogym_result, "histogym")
    validate_pair(wsi, histogym)
    curves = (wsi, histogym)
    output = output.expanduser().resolve()
    _protect_output(output, curves)
    crossover = first_crossover_episode(wsi, histogym)
    require(
        bool(np.all(histogym.full_cumulative_ns[crossover:] >= wsi.full_cumulative_ns[crossover:])),
        "HistoGym drops below WSI-Lab again after the first crossover",
    )

    temporary_parent = output.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=temporary_parent))
    try:
        csv_path = temporary / "cumulative_runtime.csv"
        png_path = temporary / "cumulative_runtime.png"
        pdf_path = temporary / "cumulative_runtime.pdf"
        readme_path = temporary / "README.md"
        _write_csv(csv_path, wsi, histogym)
        _write_plot(
            png_path,
            pdf_path,
            wsi,
            histogym,
            crossover,
            show_wsi_start_annotation=show_wsi_start_annotation,
            show_scope_subtitle=show_scope_subtitle,
        )
        _write_text(readme_path, _readme(wsi, histogym, crossover))

        selected_episodes = sorted(
            {
                0,
                1,
                10,
                50,
                100,
                500,
                1_000,
                2_000,
                crossover - 1,
                crossover,
                5_000,
                10_000,
            }
        )
        summary = {
            "schema_version": 1,
            "artifact_type": "histogym_comparison_dense_cumulative_runtime",
            "status": "complete",
            "scenario": SCENARIO,
            "plot_configuration": {
                "show_scope_subtitle": show_scope_subtitle,
                "show_wsi_start_annotation": show_wsi_start_annotation,
            },
            "semantics": {
                "x_axis": "completed episode count; episode 0 is the fixed preparation/import base",
                "y_axis": "full cumulative measured wall time",
                "formula_ns_at_episode_n": (
                    "preparation_wall_ns + import_wall_ns + "
                    "episode_cumulative_online_wall_ns[n - 1]"
                ),
                "episode_zero_formula_ns": "preparation_wall_ns + import_wall_ns",
                "dense_exact_episode_boundaries": True,
                "interpolation_used": False,
                "confidence_band": False,
                "included": [
                    "fresh scenario-specific preprocessing",
                    "measured import/setup",
                    "online outer wall time through episode N",
                ],
                "excluded": [
                    "cache-drop waiting",
                    "result/trace serialization",
                    "post-hoc tissue analysis",
                    "belief and policy inference",
                    "online embedding reads",
                    "rendering",
                    "original UNI-v2 embedding generation",
                ],
            },
            "plan": {
                "content_sha256": wsi.plan["content_sha256"],
                "file_sha256": wsi.plan["file_sha256"],
                "manifest_sha256": wsi.plan["manifest_sha256"],
                "protocol_sha256": wsi.plan["protocol_sha256"],
                "episodes": EXPECTED_EPISODES,
                "accepted_steps_per_episode": EXPECTED_ACCEPTED_STEPS_PER_EPISODE,
            },
            "timing_base_ns": {
                "wsi_lab": {
                    "preparation": wsi.preparation_ns,
                    "import": wsi.import_ns,
                    "episode_zero_total": wsi.initial_ns,
                },
                "histogym": {
                    "preparation": histogym.preparation_ns,
                    "import": histogym.import_ns,
                    "episode_zero_total": histogym.initial_ns,
                },
            },
            "crossover": {
                "definition": (
                    "first completed episode where HistoGym full cumulative wall time "
                    "is greater than or equal to WSI-Lab"
                ),
                "episode": crossover,
                "previous_episode": _point_record(crossover - 1, wsi, histogym),
                "crossover_episode": _point_record(crossover, wsi, histogym),
                "histogym_remains_higher_through_episode_10000": True,
            },
            "selected_points": [
                _point_record(episode, wsi, histogym) for episode in selected_episodes
            ],
            "final": {
                **_point_record(EXPECTED_EPISODES, wsi, histogym),
                "histogym_to_wsi_lab_full_cumulative_ratio": (
                    int(histogym.full_cumulative_ns[-1]) / int(wsi.full_cumulative_ns[-1])
                ),
                "wsi_lab_time_saved_seconds": (
                    int(histogym.full_cumulative_ns[-1]) - int(wsi.full_cumulative_ns[-1])
                )
                / 1e9,
            },
            "sources": {
                "wsi_lab": {
                    "result_sha256": wsi.result_sha256,
                    "trace_sha256": wsi.trace_sha256,
                },
                "histogym": {
                    "result_sha256": histogym.result_sha256,
                    "trace_sha256": histogym.trace_sha256,
                },
                "analyzer_sha256": analyzer_sha256,
            },
            "outputs": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in (csv_path, png_path, pdf_path, readme_path)
            },
            "measurement_isolation": {
                "post_hoc_only": True,
                "environments_called": False,
                "measured_results_modified": False,
                "derived_work_in_reported_timing": False,
            },
        }
        _write_json(temporary / "summary.json", summary)

        require(sha256_file(analyzer_path) == analyzer_sha256, "analyzer changed during generation")
        for curve in curves:
            require(
                sha256_file(curve.result_path) == curve.result_sha256,
                f"{curve.backend} result changed during generation",
            )
            require(
                sha256_file(curve.trace_path) == curve.trace_sha256,
                f"{curve.backend} trace changed during generation",
            )
        _protect_output(output, curves)
        _publish_directory_noreplace(temporary, output)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wsi-result",
        type=Path,
        default=(
            HARNESS_ROOT / "state/longrun/balanced_random_10k/wsi_lab/results/longrun_online.json"
        ),
    )
    parser.add_argument(
        "--histogym-result",
        type=Path,
        default=HARNESS_ROOT / "state/longrun/balanced_random_10k/histogym/result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HARNESS_ROOT / "outputs/longrun/cumulative_runtime_balanced_10k",
    )
    parser.add_argument(
        "--omit-wsi-start-annotation",
        action="store_true",
        help="omit the WSI-Lab episode-zero text annotation while retaining its marker",
    )
    parser.add_argument(
        "--short-title",
        action="store_true",
        help="use only the main title and omit the preprocessing/import/online subtitle",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = generate(
        args.wsi_result,
        args.histogym_result,
        args.output_dir,
        show_wsi_start_annotation=not args.omit_wsi_start_annotation,
        show_scope_subtitle=not args.short_title,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
