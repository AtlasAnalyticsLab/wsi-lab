#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly HARNESS_ROOT="${EXP_ROOT}/harness"
readonly COMPARISON_ROOT="${HARNESS_ROOT}/benchmarks/histogym_comparison"
readonly WORK_ROOT="${EXP_ROOT}/work"
readonly RUN_ROOT="${EXP_ROOT}/runs"
readonly WSI_SOURCE_ROOT="${EXP_ROOT}/runtime/wsi-lab-source"
readonly WSI_PYTHON="${EXP_ROOT}/runtime/wsi-lab-env/bin/python"
readonly CONFIG="${HARNESS_ROOT}/configs/camelyon16.yaml"
readonly MANIFEST="${WORK_ROOT}/cohort.json"
readonly PROTOCOL="${WORK_ROOT}/protocol.json"
readonly PLAN="${WORK_ROOT}/balanced_random_10k.json"

if [[ $# -ne 1 || ( "$1" != "wsi_lab" && "$1" != "histogym" ) ]]; then
  echo "usage: CACHE_DROP_CONFIRMED=1 $0 {wsi_lab|histogym}" >&2
  exit 2
fi
readonly BACKEND="$1"

read_configuration() {
  "${WSI_PYTHON}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "${WORK_ROOT}/configuration.json" "$1"
}

CAMELYON16_ROOT="$(
  read_configuration dataset_root
)"
DEVICE="$(
  read_configuration device
)"
GPU_INDEX="$(
  read_configuration gpu_index
)"
readonly CAMELYON16_ROOT DEVICE GPU_INDEX
export CAMELYON16_ROOT
export HISTOGYM_CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export WSI_LAB_SOURCE_ROOT
export PYTHONPATH="${HARNESS_ROOT}:${WSI_SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${WSI_PYTHON}" "${EXP_ROOT}/scripts/record_cache_attestation.py" "${BACKEND}"

if [[ "${BACKEND}" == "wsi_lab" ]]; then
  readonly WORKSPACE="${RUN_ROOT}/wsi_lab"
  "${WSI_PYTHON}" -c \
    'import torch, sys; device=sys.argv[1]; assert torch.cuda.is_available(); assert int(device.split(":")[1]) < torch.cuda.device_count()' \
    "${DEVICE}"
  "${WSI_PYTHON}" -m benchmarks.histogym_comparison.wsi.longrun_runner prepare \
    --plan "${PLAN}" \
    --longrun-protocol "${PROTOCOL}" \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --workspace "${WORKSPACE}" \
    --device "${DEVICE}" \
    --gpu-index "${GPU_INDEX}"
  "${WSI_PYTHON}" -m benchmarks.histogym_comparison.wsi.longrun_runner online \
    --plan "${PLAN}" \
    --longrun-protocol "${PROTOCOL}" \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --workspace "${WORKSPACE}"
else
  if [[ ! -f "${RUN_ROOT}/wsi_lab/results/longrun_online.json" ]]; then
    echo "Run the WSI-Lab block first" >&2
    exit 1
  fi
  "${COMPARISON_ROOT}/histogym/run_longrun.sh" \
    --manifest "${MANIFEST}" \
    --protocol "${PROTOCOL}" \
    --episode-plan "${PLAN}" \
    --output-dir "${RUN_ROOT}/histogym" \
    --scenario balanced_random_10k
fi
