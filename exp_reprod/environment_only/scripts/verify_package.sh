#!/usr/bin/env bash
set -euo pipefail

readonly EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${EXP_ROOT}"
sha256sum --check provenance/checksums.sha256
