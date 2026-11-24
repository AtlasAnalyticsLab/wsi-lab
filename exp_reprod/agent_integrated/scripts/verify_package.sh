#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT="$(cd "${WSI_LAB_SOURCE_ROOT:-${EXP_ROOT}/../../wsi-lab}" && pwd)"

cd "${EXP_ROOT}"
sha256sum --check provenance/checksums.sha256
python3 "${EXP_ROOT}/../scripts/verify_wsi_source.py" \
  --repository-root "${REPO_ROOT}"
