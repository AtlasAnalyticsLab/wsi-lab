#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly HARNESS_ROOT="${EXP_ROOT}/harness"
readonly WSI_SOURCE_ROOT="${EXP_ROOT}/runtime/wsi-lab-source"
readonly PYTHON="${EXP_ROOT}/runtime/wsi-lab-env/bin/python"
readonly OUTPUT_ROOT="${EXP_ROOT}/analysis_outputs"
readonly WSI_RESULT="${EXP_ROOT}/runs/wsi_lab/results/longrun_online.json"
readonly HISTOGYM_RESULT="${EXP_ROOT}/runs/histogym/result.json"

export WSI_LAB_SOURCE_ROOT
export PYTHONPATH="${HARNESS_ROOT}:${WSI_SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" "${EXP_ROOT}/scripts/summarize.py" \
  --wsi-result "${WSI_RESULT}" \
  --histogym-result "${HISTOGYM_RESULT}" \
  --output-dir "${OUTPUT_ROOT}/metrics"

"${PYTHON}" "${HARNESS_ROOT}/benchmarks/histogym_comparison/analysis/plot_cumulative_runtime.py" \
  --wsi-result "${WSI_RESULT}" \
  --histogym-result "${HISTOGYM_RESULT}" \
  --output-dir "${OUTPUT_ROOT}/cumulative_runtime" \
  --omit-wsi-start-annotation \
  --short-title
