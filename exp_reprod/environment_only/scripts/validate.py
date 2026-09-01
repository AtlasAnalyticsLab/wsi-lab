#!/usr/bin/env python3
"""Fail-fast validation without opening a WSI or running an episode."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

EXP_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = EXP_ROOT / "harness"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from benchmarks.histogym_comparison.common.longrun import load_episode_plan  # noqa: E402

WSI_REVISION = "0deb7837364add790f9530c41f0ef0b5e616c05c"
HISTOGYM_REVISION = "5b742760321d28a469d2ef9494e886a49d0aa7f7"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    work = EXP_ROOT / "work"
    runtime = EXP_ROOT / "runtime"
    source = runtime / "wsi-lab-source"
    histogym = (
        HARNESS_ROOT
        / "benchmarks/histogym_comparison/histogym/upstream/HistoGym"
    )
    wsi_python = runtime / "wsi-lab-env/bin/python"
    histogym_python = (
        HARNESS_ROOT / "benchmarks/histogym_comparison/histogym/.conda/bin/python"
    )
    required = [
        work / "cohort.json",
        work / "configuration.json",
        work / "protocol.json",
        work / "balanced_random_10k.json",
        source / "pyproject.toml",
        histogym / "gymhisto.py",
        wsi_python,
        histogym_python,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"setup/configuration is incomplete: {missing}")
    if git_head(source) != WSI_REVISION:
        raise RuntimeError("WSI-Lab source revision differs from the measured revision")
    if git_head(histogym) != HISTOGYM_REVISION:
        raise RuntimeError("HistoGym source revision differs from the measured revision")

    plan = load_episode_plan(
        work / "balanced_random_10k.json",
        work / "protocol.json",
        work / "cohort.json",
    )
    if plan.episode_count != 10_000 or plan.accepted_steps_per_episode != 100:
        raise RuntimeError("configured workload is not 10,000 x 100")
    if len(plan.unique_slide_ids) != 100:
        raise RuntimeError("configured workload does not contain 100 unique slides")

    configuration = json.loads((work / "configuration.json").read_text(encoding="utf-8"))
    device = str(configuration["device"])
    gpu_index = int(configuration["gpu_index"])
    if device != f"cuda:{gpu_index}":
        raise RuntimeError("configured CUDA device and GPU index disagree")
    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")

    manifest = json.loads((work / "cohort.json").read_text(encoding="utf-8"))
    missing_inputs: list[str] = []
    for row in [*manifest["slides"], *manifest["pilot_slides"]]:
        paths = [Path(row["wsi_path"]), Path(row["xml_path"])]
        paths.extend(
            Path(record["path"]) for record in row["metadata"]["embeddings"].values()
        )
        missing_inputs.extend(str(path) for path in paths if not path.is_file())
    if missing_inputs:
        raise FileNotFoundError(f"configured dataset inputs are missing: {missing_inputs[:8]}")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HARNESS_ROOT), str(source / "src"), env.get("PYTHONPATH", "")]
    )
    env["WSI_LAB_SOURCE_ROOT"] = str(source)
    subprocess.run(
        [str(histogym_python), str(histogym.parent.parent / "verify_setup.py")],
        check=True,
        env=env,
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "experiment": "environment_only",
                "episodes": plan.episode_count,
                "accepted_actions": plan.episode_count * plan.accepted_steps_per_episode,
                "slides": len(plan.unique_slide_ids),
                "device": device,
                "wsi_lab_revision": WSI_REVISION,
                "histogym_revision": HISTOGYM_REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
