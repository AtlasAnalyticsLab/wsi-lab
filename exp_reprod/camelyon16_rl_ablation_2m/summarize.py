#!/usr/bin/env python3
"""Entry point for verifying and summarizing newly generated study outputs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPRODUCTION_DIR = Path(__file__).resolve().parent
if str(REPRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_DIR))

from ablation.summarize import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
