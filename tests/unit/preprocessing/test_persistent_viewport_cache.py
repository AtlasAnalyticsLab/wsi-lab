from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from rl4xs_wsi.wsi.cache import PersistentViewportCache


def test_persistent_viewport_cache_is_lossless_lazy_and_input_fingerprinted(tmp_path: Path) -> None:
    slide = tmp_path / "slide.tif"
    slide.write_bytes(b"slide-v1")
    cache = PersistentViewportCache(tmp_path / "rgb")
    bounds = (0, 0, 512, 512)
    size = (8, 8)
    value = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    assert cache.get(slide, bounds, size) is None
    path = cache.put(slide, bounds, size, value)
    assert path.is_file()
    assert np.array_equal(cache.get(slide, bounds, size), value)
    assert len(tuple((tmp_path / "rgb").rglob("*.png"))) == 1

    stat = slide.stat()
    os.utime(slide, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert cache.get(slide, bounds, size) is None
