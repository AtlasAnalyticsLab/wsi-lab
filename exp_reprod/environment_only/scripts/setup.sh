#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT="$(cd "${WSI_LAB_SOURCE_ROOT:-${EXP_ROOT}/../../wsi-lab}" && pwd)"
readonly RUNTIME_ROOT="${EXP_ROOT}/runtime"
readonly WSI_SOURCE_ROOT="${REPO_ROOT}"
readonly WSI_ENV_PREFIX="${RUNTIME_ROOT}/wsi-lab-env"
readonly HISTOGYM_DIR="${EXP_ROOT}/harness/benchmarks/histogym_comparison/histogym"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH" >&2
  exit 1
fi

mkdir -p "${RUNTIME_ROOT}"
python3 "${EXP_ROOT}/../scripts/verify_wsi_source.py" \
  --repository-root "${REPO_ROOT}"

if [[ ! -x "${WSI_ENV_PREFIX}/bin/python" ]]; then
  conda env create --prefix "${WSI_ENV_PREFIX}" --file "${EXP_ROOT}/environments/wsi_lab.yml"
fi
"${WSI_ENV_PREFIX}/bin/python" -m pip install --no-deps --editable "${WSI_SOURCE_ROOT}"

"${HISTOGYM_DIR}/setup.sh"

echo "Environment-only setup complete."
echo "WSI-Lab submitted source: ${WSI_SOURCE_ROOT}"
echo "WSI-Lab Python: ${WSI_ENV_PREFIX}/bin/python"
echo "HistoGym runtime: ${HISTOGYM_DIR}/.conda/bin/python"
