"""Spatial observation projection and Gymnasium spaces."""

from __future__ import annotations

import numpy as np

from wsi_lab.wsi.reader import OverviewTransform


def make_observation_space(overview_size: int, viewport_size: int):
    from gymnasium import spaces

    return spaces.Dict(
        {
            "global": spaces.Box(0.0, 1.0, (5, overview_size, overview_size), dtype=np.float32),
            "viewport": spaces.Box(0.0, 1.0, (3, viewport_size, viewport_size), dtype=np.float32),
            "scalars": spaces.Box(0.0, 1.0, (3,), dtype=np.float32),
        }
    )


def rgb_chw_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB array, got {array.shape}")
    result = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer) or result.max(initial=0.0) > 1.0:
        result /= 255.0
    return np.clip(result.transpose(2, 0, 1), 0.0, 1.0).astype(np.float32, copy=False)


def location_mask(transform: OverviewTransform, bounds: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((transform.canvas_height, transform.canvas_width), dtype=np.float32)
    x0, y0, x1, y1 = transform.raster_bounds(bounds)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def increment_visits(
    visits: np.ndarray, transform: OverviewTransform, bounds: tuple[int, int, int, int]
) -> None:
    x0, y0, x1, y1 = transform.raster_bounds(bounds)
    region = visits[y0:y1, x0:x1]
    np.add(region, 1, out=region, casting="unsafe")
