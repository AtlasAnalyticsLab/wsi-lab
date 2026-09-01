#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly HARNESS_ROOT="${EXP_ROOT}/harness"
readonly WSI_SOURCE_ROOT="${EXP_ROOT}/runtime/wsi-lab-source"
readonly PYTHON="${EXP_ROOT}/runtime/unified-env/bin/python"
readonly OUTPUT_DIR="${EXP_ROOT}/analysis_outputs/native_observation_ppo_10k"

export WSI_LAB_SOURCE_ROOT
export PYTHONPATH="${HARNESS_ROOT}:${WSI_SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" "${HARNESS_ROOT}/benchmarks/histogym_comparison/dl_online/analyze.py" \
  --wsi-result "${EXP_ROOT}/runs/wsi_lab/online/result.json" \
  --histogym-result "${EXP_ROOT}/runs/histogym/online/result.json" \
  --output-dir "${OUTPUT_DIR}"
