#!/usr/bin/env python3
"""Create portable, hash-bound inputs for the environment-only experiment."""

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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--device",
        required=True,
        help="CUDA device used for WSI-Lab preprocessing, for example cuda:0",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not args.device.startswith("cuda:"):
        raise ValueError("--device must be an explicit CUDA device such as cuda:0")
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

        protocol = json.loads(
            (EXP_ROOT / "inputs/protocol.base.json").read_text(encoding="utf-8")
        )
        protocol["manifest"]["path"] = "cohort.json"
        protocol["manifest"]["sha256"] = sha256_file(manifest_path)
        protocol["inherits_scientific_rules_from"]["path"] = (
            "../harness/benchmarks/histogym_comparison/protocol.json"
        )
        protocol["execution"]["wsi_lab_preparation_device"] = args.device
        protocol["scenarios"] = {
            "balanced_random_10k": protocol["scenarios"]["balanced_random_10k"]
        }
        protocol["execution"]["scenario_order"] = ["balanced_random_10k"]
        protocol["execution"]["cache_drop_boundaries"] = [
            "before_wsi_lab_balanced_random_10k",
            "before_histogym_balanced_random_10k",
        ]
        protocol_path = temporary / "protocol.json"
        write_json(protocol_path, protocol)

        plan = generate_episode_plan_document(
            protocol_path, manifest_path, "balanced_random_10k"
        )
        plan_path = temporary / "balanced_random_10k.json"
        write_json(plan_path, plan)
        write_json(
            temporary / "configuration.json",
            {
                "schema_version": 1,
                "dataset_root": str(dataset_root),
                "device": args.device,
                "gpu_index": gpu_index,
                "manifest_sha256": sha256_file(manifest_path),
                "protocol_sha256": sha256_file(protocol_path),
                "plan_sha256": sha256_file(plan_path),
                "plan_content_sha256": plan["content_sha256"],
                "episodes": 10_000,
                "accepted_actions_per_episode": 100,
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
