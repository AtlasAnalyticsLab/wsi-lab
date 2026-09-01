#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_PREFIX="${SCRIPT_DIR}/.conda"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing pinned HistoGym environment: ${ENV_PREFIX}" >&2
  exit 1
fi

: "${HISTOGYM_CUDA_VISIBLE_DEVICES:?set by the reproduction wrapper}"
export CUDA_VISIBLE_DEVICES="${HISTOGYM_CUDA_VISIBLE_DEVICES}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

exec "${ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/run_longrun.py" "$@"
