from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def camelyon16_root() -> Path:
    configured_root = os.environ.get("CAMELYON16_ROOT")
    if not configured_root:
        pytest.skip("CAMELYON16_ROOT is not set")
    root = Path(configured_root)
    if not root.is_dir():
        pytest.skip(f"CAMELYON16 is unavailable: {root}")
    return root
