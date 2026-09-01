#!/usr/bin/env python3
"""Best-effort unprivileged eviction of benchmark input file pages."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from benchmarks.histogym_comparison.dl_online.common import DEFAULT_PROTOCOL, load_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("wsi_lab", "histogym"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    manifest = json.loads(protocol.manifest_path.read_text(encoding="utf-8"))
    paths: set[Path] = set()
    for row in manifest["slides"]:
        paths.add(Path(row["wsi_path"]).resolve())
        paths.add(Path(row["xml_path"]).resolve())
        if args.backend == "wsi_lab":
            for embedding in row["metadata"]["embeddings"].values():
                paths.add(Path(embedding["path"]).resolve())
    os.sync()
    started = time.perf_counter_ns()
    succeeded = 0
    bytes_advised = 0
    failures: list[dict[str, str]] = []
    advice = getattr(os, "POSIX_FADV_DONTNEED", 4)
    for path in sorted(paths):
        try:
            with path.open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, advice)
            succeeded += 1
            bytes_advised += path.stat().st_size
        except (AttributeError, OSError) as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "backend": args.backend,
        "method": "sync then posix_fadvise(POSIX_FADV_DONTNEED) on fixed input files",
        "privileged_global_drop": False,
        "requested_files": len(paths),
        "successful_files": succeeded,
        "bytes_advised": bytes_advised,
        "failures": failures,
        "wall_ns": time.perf_counter_ns() - started,
        "protocol_sha256": protocol.sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
