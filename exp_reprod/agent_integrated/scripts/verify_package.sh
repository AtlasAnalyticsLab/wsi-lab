#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT="$(cd "${EXP_ROOT}/../.." && pwd)"

cd "${EXP_ROOT}"
sha256sum --check provenance/checksums.sha256
python3 "${REPO_ROOT}/exp_reprod/scripts/verify_wsi_source.py" \
  --repository-root "${REPO_ROOT}"
