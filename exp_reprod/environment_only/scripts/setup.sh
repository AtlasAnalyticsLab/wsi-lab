#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT="$(git -C "${EXP_ROOT}" rev-parse --show-toplevel)"
readonly RUNTIME_ROOT="${EXP_ROOT}/runtime"
readonly WSI_SOURCE_ROOT="${RUNTIME_ROOT}/wsi-lab-source"
readonly WSI_ENV_PREFIX="${RUNTIME_ROOT}/wsi-lab-env"
readonly WSI_REVISION="0deb7837364add790f9530c41f0ef0b5e616c05c"
readonly HISTOGYM_DIR="${EXP_ROOT}/harness/benchmarks/histogym_comparison/histogym"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi

mkdir -p "${RUNTIME_ROOT}"
if ! git -C "${REPO_ROOT}" cat-file -e "${WSI_REVISION}^{commit}" 2>/dev/null; then
  git -C "${REPO_ROOT}" fetch origin "${WSI_REVISION}"
fi
if [[ ! -e "${WSI_SOURCE_ROOT}/.git" ]]; then
  git -C "${REPO_ROOT}" worktree add --detach "${WSI_SOURCE_ROOT}" "${WSI_REVISION}"
fi
if [[ "$(git -C "${WSI_SOURCE_ROOT}" rev-parse HEAD)" != "${WSI_REVISION}" ]]; then
  echo "Unexpected WSI-Lab revision in ${WSI_SOURCE_ROOT}" >&2
  exit 1
fi
if [[ -n "$(git -C "${WSI_SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  echo "Pinned WSI-Lab worktree has tracked modifications" >&2
  exit 1
fi

if [[ ! -x "${WSI_ENV_PREFIX}/bin/python" ]]; then
  conda env create --prefix "${WSI_ENV_PREFIX}" --file "${EXP_ROOT}/environments/wsi_lab.yml"
fi
"${WSI_ENV_PREFIX}/bin/python" -m pip install --no-deps --editable "${WSI_SOURCE_ROOT}"

"${HISTOGYM_DIR}/setup.sh"

echo "Environment-only setup complete."
echo "WSI-Lab source: ${WSI_SOURCE_ROOT}"
echo "WSI-Lab Python: ${WSI_ENV_PREFIX}/bin/python"
echo "HistoGym runtime: ${HISTOGYM_DIR}/.conda/bin/python"
