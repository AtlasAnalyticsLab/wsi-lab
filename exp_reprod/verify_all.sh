#!/usr/bin/env bash
set -euo pipefail

readonly PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${PACKAGE_ROOT}"
sha256sum --check checksums.sha256

bash "${PACKAGE_ROOT}/environment_only/scripts/verify_package.sh"
bash "${PACKAGE_ROOT}/agent_integrated/scripts/verify_package.sh"
