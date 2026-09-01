#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${EXP_ROOT}/runtime/wsi-lab-env/bin/python" \
  "${EXP_ROOT}/scripts/validate.py"
