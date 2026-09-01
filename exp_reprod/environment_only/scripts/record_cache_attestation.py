#!/usr/bin/env python3
"""Record the operator-confirmed global cache-drop boundary."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
CACHE_COMMAND = "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("wsi_lab", "histogym"))
    args = parser.parse_args()
    if os.environ.get("CACHE_DROP_CONFIRMED") != "1":
        raise RuntimeError(
            "set CACHE_DROP_CONFIRMED=1 only after running the documented cache command"
        )
    destination = EXP_ROOT / "runs" / f"cache_before_{args.backend}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    value = {
        "schema_version": 1,
        "backend": args.backend,
        "attested_at_utc": datetime.now(UTC).isoformat(),
        "operator_command": CACHE_COMMAND,
        "operator_confirmed": True,
        "controller_invoked_sudo": False,
    }
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
