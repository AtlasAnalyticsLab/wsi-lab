#!/usr/bin/env python3
"""Fail-fast validation and policy smoke test without opening a WSI."""

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

from benchmarks.histogym_comparison.dl_online.common import (  # noqa: E402
    NativeObservationActorCritic,
    load_protocol,
)

REPO_ROOT = Path(
    os.environ.get("WSI_LAB_SOURCE_ROOT", EXP_ROOT.parents[1] / "wsi-lab")
).expanduser().resolve()
WSI_SOURCE_VERIFIER = EXP_ROOT.parent / "scripts/verify_wsi_source.py"
HISTOGYM_REVISION = "5b742760321d28a469d2ef9494e886a49d0aa7f7"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_wsi_source() -> dict[str, object]:
    output = subprocess.check_output(
        [
            sys.executable,
            str(WSI_SOURCE_VERIFIER),
            "--repository-root",
            str(REPO_ROOT),
        ],
        text=True,
    )
    return dict(json.loads(output))


def main() -> int:
    work = EXP_ROOT / "work"
    runtime = EXP_ROOT / "runtime"
    source = REPO_ROOT
    histogym = (
        HARNESS_ROOT
        / "benchmarks/histogym_comparison/histogym/upstream/HistoGym"
    )
    required = [
        work / "cohort.json",
        work / "configuration.json",
        work / "balanced_random_10k.json",
        work / "protocol.json",
        source / "pyproject.toml",
        histogym / "gymhisto.py",
        runtime / "unified-env/bin/python",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"setup/configuration is incomplete: {missing}")
    wsi_source_identity = verify_wsi_source()
    if git_head(histogym) != HISTOGYM_REVISION:
        raise RuntimeError("HistoGym source revision differs from the measured revision")

    protocol = load_protocol(work / "protocol.json")
    expected_source_identity = protocol.value["source_identity"]["wsi_lab"]
    if (
        expected_source_identity["manifest_sha256"]
        != wsi_source_identity["manifest_sha256"]
    ):
        raise RuntimeError("WSI-Lab source identity differs from the configured protocol")
    if len(protocol.episodes) != 10_000 or protocol.total_interactions != 1_000_000:
        raise RuntimeError("configured workload is not 10,000 x 100")
    configuration = json.loads((work / "configuration.json").read_text(encoding="utf-8"))
    device = str(configuration["device"])
    gpu_index = int(configuration["gpu_index"])
    if device != f"cuda:{gpu_index}":
        raise RuntimeError("configured CUDA device and GPU index disagree")
    if str(protocol.value["policy"]["device"]) != device:
        raise RuntimeError("configured CUDA device differs from the PPO protocol")
    if int(protocol.value["measurement"]["gpu_index"]) != gpu_index:
        raise RuntimeError("configured GPU index differs from the PPO protocol")
    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"configured CUDA device is unavailable: {device}")
    wsi = NativeObservationActorCritic("wsi_lab", seed=1)
    histo = NativeObservationActorCritic("histogym", seed=1)
    wsi_metadata = wsi.metadata()
    histo_metadata = histo.metadata()
    for component in ("viewport_encoder", "recurrent", "critic"):
        if (
            wsi_metadata["initial_component_sha256"][component]
            != histo_metadata["initial_component_sha256"][component]
        ):
            raise RuntimeError(f"shared initial {component} weights differ")
    if wsi_metadata["trainable_parameter_count"] != 739_340:
        raise RuntimeError("WSI-Lab policy parameter count drifted")
    if histo_metadata["trainable_parameter_count"] != 555_176:
        raise RuntimeError("HistoGym policy parameter count drifted")

    print(
        json.dumps(
            {
                "status": "ready",
                "experiment": "agent_integrated",
                "episodes": len(protocol.episodes),
                "interactions": protocol.total_interactions,
                "wsi_lab_parameters": wsi_metadata["trainable_parameter_count"],
                "histogym_parameters": histo_metadata["trainable_parameter_count"],
                "shared_initial_weights_verified": True,
                "device": device,
                "wsi_lab_source_identity": wsi_source_identity,
                "histogym_revision": HISTOGYM_REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
