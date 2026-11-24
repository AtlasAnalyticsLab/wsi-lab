#!/usr/bin/env python3
"""Create the five primary efficiency metrics from completed long-run results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complete(path: Path, backend: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("scenario") != "balanced_random_10k":
        raise ValueError(f"incomplete or incorrect {backend} result: {path}")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or checkpoints[-1].get("accepted_steps") != 1_000_000:
        raise ValueError(f"{backend} result does not contain one million accepted actions")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-result", type=Path, required=True)
    parser.add_argument("--histogym-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    results = {
        "wsi_lab": load_complete(args.wsi_result.resolve(), "WSI-Lab"),
        "histogym": load_complete(args.histogym_result.resolve(), "HistoGym"),
    }
    final = {name: value["checkpoints"][-1] for name, value in results.items()}
    if final["wsi_lab"]["plan_content_sha256"] != final["histogym"]["plan_content_sha256"]:
        raise ValueError("backend results use different episode plans")

    def seconds(backend: str, key: str) -> float:
        return float(final[backend]["components_ns"][key]) / 1e9

    rows = [
        {
            "metric": "Preprocessing runtime",
            "unit": "s",
            "wsi_lab": seconds("wsi_lab", "preparation_wall"),
            "histogym": seconds("histogym", "preparation_wall"),
            "speedup": None,
        },
        {
            "metric": "Online runtime",
            "unit": "s",
            "wsi_lab": seconds("wsi_lab", "cumulative_online_outer_wall"),
            "histogym": seconds("histogym", "cumulative_online_outer_wall"),
            "speedup": seconds("histogym", "cumulative_online_outer_wall")
            / seconds("wsi_lab", "cumulative_online_outer_wall"),
        },
        {
            "metric": "End-to-end runtime",
            "unit": "s",
            "wsi_lab": seconds("wsi_lab", "full_preparation_import_online_wall"),
            "histogym": seconds("histogym", "full_preparation_import_online_wall"),
            "speedup": seconds("histogym", "full_preparation_import_online_wall")
            / seconds("wsi_lab", "full_preparation_import_online_wall"),
        },
        {
            "metric": "Native step throughput",
            "unit": "actions/s",
            "wsi_lab": float(
                final["wsi_lab"]["throughput"]["accepted_steps_per_second_step_calls"]
            ),
            "histogym": float(
                final["histogym"]["throughput"]["accepted_steps_per_second_step_calls"]
            ),
            "speedup": float(
                final["wsi_lab"]["throughput"]["accepted_steps_per_second_step_calls"]
            )
            / float(final["histogym"]["throughput"]["accepted_steps_per_second_step_calls"]),
        },
        {
            "metric": "Overall online throughput",
            "unit": "actions/s",
            "wsi_lab": float(
                final["wsi_lab"]["throughput"]["accepted_steps_per_second_online_outer"]
            ),
            "histogym": float(
                final["histogym"]["throughput"]["accepted_steps_per_second_online_outer"]
            ),
            "speedup": float(
                final["wsi_lab"]["throughput"]["accepted_steps_per_second_online_outer"]
            )
            / float(
                final["histogym"]["throughput"]["accepted_steps_per_second_online_outer"]
            ),
        },
    ]
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "environment_only",
        "one_measured_execution_per_backend": True,
        "workload": {"slides": 100, "episodes": 10_000, "accepted_actions": 1_000_000},
        "plan_content_sha256": final["wsi_lab"]["plan_content_sha256"],
        "metrics": rows,
        "sources": {
            "wsi_lab_sha256": sha256_file(args.wsi_result),
            "histogym_sha256": sha256_file(args.histogym_result),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Environment-only efficiency comparison",
        "",
        "One measured execution per backend; 10,000 episodes and 1,000,000 accepted actions.",
        "",
        "| Metric | WSI-Lab | HistoGym | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        speedup = "NA" if row["speedup"] is None else f"{row['speedup']:.4f}x"
        lines.append(
            f"| {row['metric']} ({row['unit']}) | {row['wsi_lab']:.4f} | "
            f"{row['histogym']:.4f} | {speedup} |"
        )
    lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "metrics": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
