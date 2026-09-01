#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly UPSTREAM_DIR="${SCRIPT_DIR}/upstream/HistoGym"
readonly ENV_PREFIX="${SCRIPT_DIR}/.conda"
readonly HISTOGYM_URL="https://github.com/xjtuai/histogym.git"
readonly HISTOGYM_COMMIT="5b742760321d28a469d2ef9494e886a49d0aa7f7"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${UPSTREAM_DIR}")"
  git clone --no-checkout "${HISTOGYM_URL}" "${UPSTREAM_DIR}"
fi

if [[ -n "$(git -C "${UPSTREAM_DIR}" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to change a tracked, dirty HistoGym checkout: ${UPSTREAM_DIR}" >&2
  exit 1
fi

if ! git -C "${UPSTREAM_DIR}" cat-file -e "${HISTOGYM_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${UPSTREAM_DIR}" fetch --no-tags origin "${HISTOGYM_COMMIT}"
fi
git -C "${UPSTREAM_DIR}" checkout --detach "${HISTOGYM_COMMIT}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda env create --prefix "${ENV_PREFIX}" --file "${SCRIPT_DIR}/environment.yml"
fi

"${ENV_PREFIX}/bin/python" "${SCRIPT_DIR}/verify_setup.py"
