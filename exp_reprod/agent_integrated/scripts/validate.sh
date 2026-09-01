#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly WSI_SOURCE_ROOT="${EXP_ROOT}/runtime/wsi-lab-source"

export WSI_LAB_SOURCE_ROOT="${WSI_SOURCE_ROOT}"
export PYTHONPATH="${EXP_ROOT}/harness:${WSI_SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${EXP_ROOT}/runtime/unified-env/bin/python" \
  "${EXP_ROOT}/scripts/validate.py"
