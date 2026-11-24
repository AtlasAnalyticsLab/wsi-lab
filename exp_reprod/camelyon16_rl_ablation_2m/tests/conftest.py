"""Import setup for package-local reproduction tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPRODUCTION_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(
    os.environ.get("WSI_LAB_SOURCE_ROOT", REPRODUCTION_DIR.parents[1] / "wsi-lab")
).expanduser().resolve()
for path in (REPRODUCTION_DIR, REPOSITORY_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
