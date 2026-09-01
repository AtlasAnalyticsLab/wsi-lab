#!/usr/bin/env python3
"""Validate and report the two measured DL-online experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

from benchmarks.histogym_comparison.dl_online.common import sha256_file


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"incomplete result: {path}")
    if value.get("pilot_excluded"):
        raise ValueError(f"pilot result cannot enter measured analysis: {path}")
    return value


def _trace(result: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(result["artifacts"]["trace"])
    if sha256_file(path) != result["artifacts"]["trace_sha256"]:
        raise ValueError(f"trace hash drifted: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def _validations(results: dict[str, dict[str, Any]], traces: dict[str, dict[str, np.ndarray]]):
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise ValueError(f"validation failed: {name}: {detail}")

    histogym = results["histogym"]
    wsi_lab = results["wsi_lab"]
    check(
        "same_protocol",
        histogym["protocol"]["sha256"] == wsi_lab["protocol"]["sha256"],
        "both results bind the same DL-online protocol",
    )
    for backend, result in results.items():
        workload = result["workload"]
        check(
            f"{backend}_fixed_workload",
            workload["episode_count"] == 10_000
            and workload["total_interactions"] == 1_000_000
            and workload["interactions_per_episode"] == 100,
            "10,000 episodes x 100 interactions",
        )
        check(
            f"{backend}_trace_length",
            len(traces[backend]["step_ns"]) == 1_000_000,
            "trace contains one row per interaction",
        )
        check(
            f"{backend}_ppo_updates",
            result["policy"]["ppo_update_count"] == 1_250,
            "800-transition rollouts produce exactly 1,250 updates",
        )
    for component in ("viewport_encoder", "recurrent", "critic"):
        left = histogym["policy"]["initial"]["initial_component_sha256"][component]
        right = wsi_lab["policy"]["initial"]["initial_component_sha256"][component]
        check(
            f"shared_initial_{component}",
            left == right,
            "shared-shape component starts bit-identically",
        )
    check(
        "histogym_stay_masked",
        int(histogym["actions"]["6"]) == 0,
        "HistoGym stay action was never selected",
    )
    check(
        "wsi_diagnosis_masked",
        int(wsi_lab["actions"]["9"]) == 0 and int(wsi_lab["actions"]["10"]) == 0,
        "WSI-Lab diagnosis actions were never selected",
    )
    belief = wsi_lab["belief"]
    check(
        "frozen_belief_called",
        int(belief["belief_add_calls"]) > 0
        and int(belief["belief_add_calls"]) == int(belief["embedding_reads"]),
        "each native evidence addition used one precomputed embedding read",
    )
    check(
        "uni_generation_excluded",
        not bool(belief["uni_v2_generation_included"]),
        "UNI-v2 generation is outside the measured experiment",
    )
    return checks


def _metric_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        key: str,
        label: str,
        unit: str,
        getter,
        *,
        lower_is_better: bool,
    ) -> None:
        values = {backend: getter(result) for backend, result in results.items()}
        hist = values["histogym"]
        wsi = values["wsi_lab"]
        ratio = None if hist in {None, 0} or wsi in {None, 0} else float(hist) / float(wsi)
        rows.append(
            {
                "key": key,
                "metric": label,
                "unit": unit,
                "wsi_lab": wsi,
                "histogym": hist,
                "histogym_over_wsi_lab": ratio,
                "lower_is_better": lower_is_better,
            }
        )

    def ns_s(value: int) -> float:
        return value / 1e9

    add(
        "parameters",
        "Trainable policy parameters",
        "count",
        lambda r: r["policy"]["initial"]["trainable_parameter_count"],
        lower_is_better=True,
    )
    add(
        "preprocessing",
        "Preprocessing runtime",
        "s",
        lambda r: ns_s(r["components_ns"]["preprocessing"]),
        lower_is_better=True,
    )
    add(
        "online",
        "Online outer runtime",
        "s",
        lambda r: ns_s(r["components_ns"]["online_outer"]),
        lower_is_better=True,
    )
    add(
        "end_to_end",
        "End-to-end runtime",
        "s",
        lambda r: ns_s(r["components_ns"]["end_to_end"]),
        lower_is_better=True,
    )
    for key, label in (
        ("native_step", "Native environment steps"),
        ("policy_forward_and_action", "Policy inference and action sampling"),
        ("ppo_optimization", "PPO optimization"),
    ):
        add(
            key,
            label,
            "s",
            lambda r, key=key: ns_s(r["components_ns"][key]),
            lower_is_better=True,
        )
    add(
        "native_step_median",
        "Native step median latency",
        "ms",
        lambda r: r["latency"]["native_step"]["median_ms"],
        lower_is_better=True,
    )
    add(
        "native_step_p95",
        "Native step p95 latency",
        "ms",
        lambda r: r["latency"]["native_step"]["p95_ms"],
        lower_is_better=True,
    )
    add(
        "native_throughput",
        "Native step throughput",
        "interactions/s",
        lambda r: r["throughput"]["native_step_calls_per_second"],
        lower_is_better=False,
    )
    add(
        "overall_online_throughput",
        "Overall online throughput",
        "interactions/s",
        lambda r: r["throughput"]["overall_online_interactions_per_second"],
        lower_is_better=False,
    )
    add(
        "rss",
        "Peak process-tree RSS",
        "MiB",
        lambda r: r["resources"]["peak_rss_bytes"] / 2**20,
        lower_is_better=True,
    )
    add(
        "gpu_memory",
        "Incremental peak GPU memory",
        "MiB",
        lambda r: r["resources"]["gpu_peak_incremental_memory_mib"],
        lower_is_better=True,
    )
    add(
        "cpu",
        "Average process-tree CPU",
        "% (one core=100)",
        lambda r: r["resources"]["average_cpu_percent_one_core_100"],
        lower_is_better=False,
    )
    add(
        "read_bytes",
        "Online process-tree physical reads",
        "GiB",
        lambda r: r["resources"]["io"]["read_bytes"] / 2**30,
        lower_is_better=True,
    )
    add(
        "observation_payload",
        "Native observation payload",
        "GiB",
        lambda r: r["observations"]["reset_plus_post_step_payload_bytes"] / 2**30,
        lower_is_better=True,
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cumulative_plot(
    output_dir: Path,
    results: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    series: dict[str, np.ndarray] = {}
    for backend, result in results.items():
        offset = int(result["components_ns"]["preprocessing"]) + int(
            result["components_ns"]["policy_model_and_optimizer_setup"]
        )
        series[backend] = np.concatenate(
            (
                np.asarray([offset], dtype=np.int64),
                offset + traces[backend]["episode_cumulative_online_ns"],
            )
        )
    episodes = np.arange(len(series["wsi_lab"]), dtype=np.int64)
    difference = series["histogym"] - series["wsi_lab"]
    crossings = np.flatnonzero(difference >= 0)
    crossover = int(crossings[0]) if crossings.size else None
    wsi_minutes = series["wsi_lab"].astype(np.float64) / 60e9
    histogym_minutes = series["histogym"].astype(np.float64) / 60e9

    # Keep this presentation synchronized with the clean environment-only
    # cumulative-runtime figure in analysis/plot_cumulative_runtime.py.
    figure, axis = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    axis.plot(episodes, wsi_minutes, color="#0072B2", linewidth=2.2, label="WSI-Lab")
    axis.plot(
        episodes,
        histogym_minutes,
        color="#D55E00",
        linewidth=2.2,
        label="HistoGym",
    )
    if crossover is not None:
        crossing_minutes = float(
            (wsi_minutes[crossover] + histogym_minutes[crossover]) / 2
        )
        axis.axvline(
            crossover,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        axis.scatter(
            [crossover, crossover],
            [wsi_minutes[crossover], histogym_minutes[crossover]],
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
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "#BBBBBB",
                "alpha": 0.9,
            },
        )
    axis.scatter([0], [wsi_minutes[0]], color="#0072B2", s=25, zorder=4)
    axis.annotate(
        f"{histogym_minutes[-1]:.1f} min",
        xy=(episodes[-1], histogym_minutes[-1]),
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
    axis.set_title("Cumulative runtime over 10,000 episodes", fontsize=12)
    axis.set_xlabel("Episode number")
    axis.set_ylabel("Cumulative wall time (minutes)")
    axis.set_xlim(0, 10_000)
    axis.set_ylim(bottom=0)
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"cumulative_runtime.{suffix}",
            **(
                {"dpi": 300}
                if suffix == "png"
                else {"metadata": {"CreationDate": None, "ModDate": None}}
            ),
        )
    plt.close(figure)
    return {
        "crossover_episode_histogym_reaches_wsi_lab": crossover,
        "episode_0_seconds": {
            backend: float(values[0] / 1e9) for backend, values in series.items()
        },
        "episode_10000_seconds": {
            backend: float(values[-1] / 1e9) for backend, values in series.items()
        },
    }


def _component_plot(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    components = [
        (("native_step",), "Environment step"),
        (
            (
                "backend_import_and_setup",
                "native_construction",
                "native_reset",
                "native_close",
            ),
            "Environment lifecycle",
        ),
        (
            ("observation_pack", "truncation_observation_pack"),
            "Observation packing",
        ),
        (
            ("policy_forward_and_action", "truncation_value_forward"),
            "Policy inference",
        ),
        (("rollout_bookkeeping", "rollout_buffer_allocation"), "Bookkeeping"),
        (("ppo_optimization",), "PPO optimization"),
        (("online_other_residual",), "Other online"),
    ]
    backends = ("wsi_lab", "histogym")
    x = np.arange(len(backends))
    bottoms = np.zeros(2)
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for keys, label in components:
        values = np.asarray(
            [
                sum(results[backend]["components_ns"][key] for key in keys) / 1e9
                for backend in backends
            ]
        )
        axis.bar(x, values, bottom=bottoms, label=label)
        bottoms += values
    axis.set_xticks(x, ("WSI-Lab", "HistoGym"))
    axis.set_ylabel("Cumulative online wall time (s)")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"online_components.{suffix}",
            dpi=180,
            **(
                {}
                if suffix == "png"
                else {"metadata": {"CreationDate": None, "ModDate": None}}
            ),
        )
    plt.close(figure)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.4f}"


def _report(
    rows: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    cumulative: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Native-observation recurrent-PPO efficiency comparison",
        "",
        f"**PASS:** {len(validations)} protocol and accounting checks passed. The result is a computational-efficiency comparison only; the all-tumor cohort and different native rewards are not used for accuracy or sample-efficiency claims.",
        "",
        "Both runs contain 10,000 balanced episodes and exactly 1,000,000 interactions. They use the same CNN topology for every visual branch, bit-identical initial weights for the shared 3-channel CNN/GRU/critic, and the same recurrent-PPO updater and hyperparameters. WSI-Lab additionally processes its native five-channel global observation and five scalars and calls the frozen belief model on precomputed embeddings.",
        "",
        "| Metric | WSI-Lab | HistoGym | HistoGym / WSI-Lab |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} ({row['unit']}) | {_fmt(row['wsi_lab'])} | {_fmt(row['histogym'])} | {_fmt(row['histogym_over_wsi_lab'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen belief cost inside WSI-Lab",
            "",
            f"WSI-Lab performed {results['wsi_lab']['belief']['embedding_reads']:,} precomputed embedding reads. Embedding lookup consumed {results['wsi_lab']['components_ns']['embedding_lookup_nested_in_native_step'] / 1e9:.3f} s and frozen ABMIL inference consumed {results['wsi_lab']['components_ns']['frozen_belief_nested_in_native_step'] / 1e9:.3f} s. These are nested within native environment-step time and are not added again to online wall time.",
            "",
            "## Cumulative end-to-end runtime",
            "",
            f"Episode 0 contains preprocessing plus policy setup: WSI-Lab {cumulative['episode_0_seconds']['wsi_lab']:.3f} s; HistoGym {cumulative['episode_0_seconds']['histogym']:.3f} s.",
            f"At episode 10,000: WSI-Lab {cumulative['episode_10000_seconds']['wsi_lab']:.3f} s; HistoGym {cumulative['episode_10000_seconds']['histogym']:.3f} s.",
            "",
            "The original UNI-v2 generation is excluded. WSI-Lab preprocessing is freshly measured and kept separate; precomputed embedding lookup and frozen ABMIL inference are included online. Result serialization and final checkpoint writes are excluded from timed/resource-monitored intervals.",
        ]
    )
    crossover = cumulative["crossover_episode_histogym_reaches_wsi_lab"]
    lines.append(
        "HistoGym does not reach WSI-Lab's cumulative time within 10,000 episodes."
        if crossover is None
        else f"HistoGym first reaches or exceeds WSI-Lab's cumulative time at episode {crossover:,}."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-result", type=Path, required=True)
    parser.add_argument("--histogym-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {
        "wsi_lab": _load(args.wsi_result.resolve()),
        "histogym": _load(args.histogym_result.resolve()),
    }
    traces = {backend: _trace(result) for backend, result in results.items()}
    validations = _validations(results, traces)
    rows = _metric_rows(results)
    cumulative = _cumulative_plot(args.output_dir, results, traces)
    _component_plot(args.output_dir, results)
    _write_csv(args.output_dir / "metrics.csv", rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "validations": validations,
        "metrics": rows,
        "cumulative_runtime": cumulative,
        "sources": {
            "wsi_lab": {
                "sha256": sha256_file(args.wsi_result),
            },
            "histogym": {
                "sha256": sha256_file(args.histogym_result),
            },
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        _report(rows, validations, cumulative, results), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "checks": len(validations)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
