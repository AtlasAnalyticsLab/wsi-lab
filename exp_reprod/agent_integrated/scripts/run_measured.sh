#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT="$(cd "${WSI_LAB_SOURCE_ROOT:-${EXP_ROOT}/../../wsi-lab}" && pwd)"
readonly HARNESS_ROOT="${EXP_ROOT}/harness"
readonly WORK_ROOT="${EXP_ROOT}/work"
readonly RUN_ROOT="${EXP_ROOT}/runs"
readonly WSI_SOURCE_ROOT="${REPO_ROOT}"
readonly PYTHON="${EXP_ROOT}/runtime/unified-env/bin/python"
readonly CONFIG="${HARNESS_ROOT}/configs/camelyon16.yaml"
readonly MANIFEST="${WORK_ROOT}/cohort.json"
readonly PROTOCOL="${WORK_ROOT}/protocol.json"

if [[ $# -ne 1 || ( "$1" != "wsi_lab" && "$1" != "histogym" ) ]]; then
  echo "usage: $0 {wsi_lab|histogym}" >&2
  exit 2
fi
readonly BACKEND="$1"

read_configuration() {
  "${PYTHON}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "${WORK_ROOT}/configuration.json" "$1"
}

CAMELYON16_ROOT="$(read_configuration dataset_root)"
DEVICE="$(read_configuration device)"
GPU_INDEX="$(read_configuration gpu_index)"
readonly CAMELYON16_ROOT DEVICE GPU_INDEX
export CAMELYON16_ROOT
export WSI_LAB_SOURCE_ROOT="${WSI_SOURCE_ROOT}"
export PYTHONPATH="${HARNESS_ROOT}:${WSI_SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_ROOT}"

if [[ "${BACKEND}" == "wsi_lab" ]]; then
  "${PYTHON}" "${HARNESS_ROOT}/benchmarks/histogym_comparison/dl_online/evict_cache.py" \
    wsi_lab --protocol "${PROTOCOL}" --output "${RUN_ROOT}/cache_before_wsi_lab.json"
  "${PYTHON}" -m benchmarks.histogym_comparison.wsi.runner prepare \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --workspace "${RUN_ROOT}/wsi_lab" \
    --device "${DEVICE}" \
    --gpu-index "${GPU_INDEX}"
  "${PYTHON}" "${HARNESS_ROOT}/benchmarks/histogym_comparison/dl_online/run.py" \
    wsi_lab \
    --protocol "${PROTOCOL}" \
    --config "${CONFIG}" \
    --workspace "${RUN_ROOT}/wsi_lab" \
    --output-dir "${RUN_ROOT}/wsi_lab/online"
else
  if [[ ! -f "${RUN_ROOT}/wsi_lab/online/result.json" ]]; then
    echo "Run the WSI-Lab block first" >&2
    exit 1
  fi
  "${PYTHON}" "${HARNESS_ROOT}/benchmarks/histogym_comparison/dl_online/evict_cache.py" \
    histogym --protocol "${PROTOCOL}" --output "${RUN_ROOT}/cache_before_histogym.json"
  "${PYTHON}" "${HARNESS_ROOT}/benchmarks/histogym_comparison/dl_online/run.py" \
    histogym \
    --protocol "${PROTOCOL}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}/histogym/online"
fi
