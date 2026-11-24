#!/usr/bin/env python3
"""Create portable, hash-bound inputs for the agent-integrated experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = EXP_ROOT / "harness"
COMPARISON_ROOT = HARNESS_ROOT / "benchmarks/histogym_comparison"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from benchmarks.histogym_comparison.common.longrun import (  # noqa: E402
    generate_episode_plan_document,
    sha256_file,
)

EXPECTED_BELIEF_SHA256 = "71fb6677a5307a53424e84a111e5b88c426e845065b85f8bf09c03f7d0d239fc"
EXPECTED_BELIEF_BYTES = 4_215_860


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--belief-checkpoint",
        type=Path,
        default=EXP_ROOT / "models/calibrated_checkpoint.pt",
        help="frozen calibrated checkpoint (defaults to the bundled checkpoint)",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="CUDA device used for preprocessing, PPO, and monitoring, for example cuda:0",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    belief_checkpoint = args.belief_checkpoint.expanduser().resolve()
    if not belief_checkpoint.is_file():
        raise FileNotFoundError(belief_checkpoint)
    belief_sha256 = sha256_file(belief_checkpoint)
    if belief_sha256 != EXPECTED_BELIEF_SHA256:
        raise ValueError(
            f"belief checkpoint hash differs: {belief_sha256} != {EXPECTED_BELIEF_SHA256}"
        )
    if belief_checkpoint.stat().st_size != EXPECTED_BELIEF_BYTES:
        raise ValueError("belief checkpoint size differs from the measured checkpoint")
    if not args.device.startswith("cuda:"):
        raise ValueError("the protocol requires an explicit CUDA device such as cuda:0")
    gpu_index = int(args.device.split(":", 1)[1])
    if gpu_index < 0:
        raise ValueError("CUDA device index must be non-negative")

    output_root = EXP_ROOT / "work"
    if output_root.exists():
        raise FileExistsError(
            f"configured input directory already exists: {output_root}; "
            "move it aside before configuring a new dataset"
        )
    temporary = Path(tempfile.mkdtemp(prefix=".work.", dir=EXP_ROOT))
    try:
        manifest_path = temporary / "cohort.json"
        generator = COMPARISON_ROOT / "common/generate_manifest.py"
        subprocess.run(
            [
                sys.executable,
                str(generator),
                "--dataset-root",
                str(dataset_root),
                "--output",
                str(manifest_path),
                "--seed",
                "1",
                "--measured-size",
                "100",
                "--expected-population-size",
                "111",
            ],
            check=True,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        definition = json.loads(
            (EXP_ROOT / "inputs/cohort_definition.json").read_text(encoding="utf-8")
        )
        for key in (
            "population_ids_sha256",
            "measured_ids_sha256",
            "pilot_ids_sha256",
        ):
            observed = manifest["sampling"][key]
            expected = definition["sampling"][key]
            if observed != expected:
                raise ValueError(f"cohort {key} differs: {observed} != {expected}")
        if manifest["reference_table"]["sha256"] != definition["reference_table_sha256"]:
            raise ValueError("CAMELYON16 reference.csv hash differs from the measured input")

        schedule_protocol = json.loads(
            (EXP_ROOT / "inputs/schedule_protocol.base.json").read_text(encoding="utf-8")
        )
        schedule_protocol["manifest"]["path"] = "cohort.json"
        schedule_protocol["manifest"]["sha256"] = sha256_file(manifest_path)
        schedule_protocol["scenarios"] = {
            "balanced_random_10k": schedule_protocol["scenarios"]["balanced_random_10k"]
        }
        schedule_protocol["execution"]["scenario_order"] = ["balanced_random_10k"]
        schedule_protocol["execution"]["cache_drop_boundaries"] = []
        schedule_protocol["execution"]["wsi_lab_preparation_device"] = args.device
        schedule_protocol_path = temporary / "schedule_protocol.json"
        write_json(schedule_protocol_path, schedule_protocol)
        schedule = generate_episode_plan_document(
            schedule_protocol_path, manifest_path, "balanced_random_10k"
        )
        schedule_path = temporary / "balanced_random_10k.json"
        write_json(schedule_path, schedule)

        protocol = json.loads(
            (EXP_ROOT / "inputs/protocol.base.json").read_text(encoding="utf-8")
        )
        protocol["cohort"]["manifest"] = "cohort.json"
        protocol["cohort"]["manifest_sha256"] = sha256_file(manifest_path)
        protocol["workload"]["slide_schedule"] = "balanced_random_10k.json"
        protocol["workload"]["slide_schedule_sha256"] = sha256_file(schedule_path)
        protocol["belief"]["checkpoint"] = str(belief_checkpoint)
        protocol["belief"]["checkpoint_sha256"] = belief_sha256
        protocol["policy"]["device"] = args.device
        protocol["measurement"]["gpu_index"] = gpu_index
        protocol_path = temporary / "protocol.json"
        write_json(protocol_path, protocol)
        write_json(
            temporary / "configuration.json",
            {
                "schema_version": 1,
                "dataset_root": str(dataset_root),
                "belief_checkpoint": str(belief_checkpoint),
                "belief_checkpoint_sha256": belief_sha256,
                "device": args.device,
                "gpu_index": gpu_index,
                "manifest_sha256": sha256_file(manifest_path),
                "schedule_protocol_sha256": sha256_file(schedule_protocol_path),
                "schedule_sha256": sha256_file(schedule_path),
                "schedule_content_sha256": schedule["content_sha256"],
                "protocol_sha256": sha256_file(protocol_path),
                "episodes": 10_000,
                "interactions_per_episode": 100,
            },
        )
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({"status": "configured", "output": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
