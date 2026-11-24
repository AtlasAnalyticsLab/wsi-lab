#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT="$(cd "${WSI_LAB_SOURCE_ROOT:-${EXP_ROOT}/../../wsi-lab}" && pwd)"
readonly RUNTIME_ROOT="${EXP_ROOT}/runtime"
readonly WSI_SOURCE_ROOT="${REPO_ROOT}"
readonly ENV_PREFIX="${RUNTIME_ROOT}/unified-env"
readonly HISTOGYM_REVISION="5b742760321d28a469d2ef9494e886a49d0aa7f7"
readonly HISTOGYM_ROOT="${EXP_ROOT}/harness/benchmarks/histogym_comparison/histogym/upstream/HistoGym"

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

if [[ ! -d "${HISTOGYM_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${HISTOGYM_ROOT}")"
  git clone --no-checkout https://github.com/xjtuai/histogym.git "${HISTOGYM_ROOT}"
fi
if ! git -C "${HISTOGYM_ROOT}" cat-file -e "${HISTOGYM_REVISION}^{commit}" 2>/dev/null; then
  git -C "${HISTOGYM_ROOT}" fetch --no-tags origin "${HISTOGYM_REVISION}"
fi
if [[ -n "$(git -C "${HISTOGYM_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  echo "Pinned HistoGym checkout has tracked modifications" >&2
  exit 1
fi
git -C "${HISTOGYM_ROOT}" checkout --detach "${HISTOGYM_REVISION}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda env create --prefix "${ENV_PREFIX}" --file "${EXP_ROOT}/environments/environment.yml"
fi
"${ENV_PREFIX}/bin/python" -m pip install --no-deps --editable "${WSI_SOURCE_ROOT}"

echo "Agent-integrated setup complete."
echo "WSI-Lab submitted source: ${WSI_SOURCE_ROOT}"
echo "HistoGym source: ${HISTOGYM_ROOT}"
echo "Unified Python: ${ENV_PREFIX}/bin/python"
