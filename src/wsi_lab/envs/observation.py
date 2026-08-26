"""Spatial observation projection and Gymnasium spaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from wsi_lab.wsi.reader import OverviewTransform

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ROIObservationTransform:
    """Map canonical level-0 coordinates into one ROI-relative canvas.

    ``global_bounds`` is the nominal snapped navigation ROI.  It deliberately
    is not clipped to the physical WSI dimensions: a snapped right or bottom
    edge may overhang the slide.  All spatial observation channels therefore
    share this exact extent while navigation cells and embeddings retain their
    original global level-0 coordinates.
    """

    global_bounds: tuple[int, int, int, int]
    canvas_width: int
    canvas_height: int

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.global_bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"ROI bounds must have positive area, got {self.global_bounds}")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("canvas dimensions must be positive")

    @classmethod
    def fit(
        cls,
        global_bounds: tuple[int, int, int, int],
        canvas_size: tuple[int, int],
    ) -> ROIObservationTransform:
        """Create the existing aspect-preserving, centered letterbox map."""

        bounds = tuple(map(int, global_bounds))
        if len(bounds) != 4:
            raise ValueError(f"Expected four ROI bounds, got {global_bounds}")
        return cls(
            global_bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
            canvas_width=int(canvas_size[0]),
            canvas_height=int(canvas_size[1]),
        )

    @property
    def global_x0(self) -> int:
        return self.global_bounds[0]

    @property
    def global_y0(self) -> int:
        return self.global_bounds[1]

    @property
    def global_x1(self) -> int:
        return self.global_bounds[2]

    @property
    def global_y1(self) -> int:
        return self.global_bounds[3]

    @property
    def source_width(self) -> int:
        """Width of the nominal ROI in canonical level-0 pixels."""

        return self.global_x1 - self.global_x0

    @property
    def source_height(self) -> int:
        """Height of the nominal ROI in canonical level-0 pixels."""

        return self.global_y1 - self.global_y0

    @property
    def content_width(self) -> int:
        scale = min(
            self.canvas_width / self.source_width,
            self.canvas_height / self.source_height,
        )
        return min(self.canvas_width, max(1, round(self.source_width * scale)))

    @property
    def content_height(self) -> int:
        scale = min(
            self.canvas_width / self.source_width,
            self.canvas_height / self.source_height,
        )
        return min(self.canvas_height, max(1, round(self.source_height * scale)))

    @property
    def content_x0(self) -> int:
        return (self.canvas_width - self.content_width) // 2

    @property
    def content_y0(self) -> int:
        return (self.canvas_height - self.content_height) // 2

    @property
    def scale_x(self) -> float:
        return self.content_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.content_height / self.source_height

    def global_to_roi_point(self, x: float, y: float) -> tuple[float, float]:
        """Translate a global point into ROI-relative level-0 units."""

        return x - self.global_x0, y - self.global_y0

    def roi_to_global_point(self, x: float, y: float) -> tuple[float, float]:
        """Translate an ROI-relative point back to canonical level-0 units."""

        return x + self.global_x0, y + self.global_y0

    def global_to_roi_bounds(self, bounds: Bounds) -> Bounds:
        x0, y0 = self.global_to_roi_point(bounds[0], bounds[1])
        x1, y1 = self.global_to_roi_point(bounds[2], bounds[3])
        return x0, y0, x1, y1

    def roi_to_global_bounds(self, bounds: Bounds) -> Bounds:
        x0, y0 = self.roi_to_global_point(bounds[0], bounds[1])
        x1, y1 = self.roi_to_global_point(bounds[2], bounds[3])
        return x0, y0, x1, y1

    def global_to_normalized_point(self, x: float, y: float) -> tuple[float, float]:
        """Normalize against the ROI edges, independently of letterboxing."""

        relative_x, relative_y = self.global_to_roi_point(x, y)
        return relative_x / self.source_width, relative_y / self.source_height

    def normalized_to_global_point(self, x: float, y: float) -> tuple[float, float]:
        return self.roi_to_global_point(x * self.source_width, y * self.source_height)

    def global_to_normalized_bounds(self, bounds: Bounds) -> Bounds:
        x0, y0 = self.global_to_normalized_point(bounds[0], bounds[1])
        x1, y1 = self.global_to_normalized_point(bounds[2], bounds[3])
        return x0, y0, x1, y1

    def normalized_to_global_bounds(self, bounds: Bounds) -> Bounds:
        x0, y0 = self.normalized_to_global_point(bounds[0], bounds[1])
        x1, y1 = self.normalized_to_global_point(bounds[2], bounds[3])
        return x0, y0, x1, y1

    def project_point(self, x: float, y: float) -> tuple[float, float]:
        """Project a global level-0 point onto the letterboxed canvas."""

        relative_x, relative_y = self.global_to_roi_point(x, y)
        return (
            self.content_x0 + relative_x * self.scale_x,
            self.content_y0 + relative_y * self.scale_y,
        )

    def unproject_point(self, x: float, y: float) -> tuple[float, float]:
        """Map a canvas point back into canonical global level-0 units."""

        return self.roi_to_global_point(
            (x - self.content_x0) / self.scale_x,
            (y - self.content_y0) / self.scale_y,
        )

    def project_bounds(self, bounds: Bounds) -> Bounds:
        x0, y0 = self.project_point(bounds[0], bounds[1])
        x1, y1 = self.project_point(bounds[2], bounds[3])
        return x0, y0, x1, y1

    def raster_bounds(
        self,
        bounds: Bounds,
        *,
        ensure_nonempty: bool = True,
    ) -> tuple[int, int, int, int]:
        """Rasterize global bounds using the same convention as overviews."""

        x0f, y0f, x1f, y1f = self.project_bounds(bounds)
        x0 = max(0, min(self.canvas_width, math.floor(x0f)))
        y0 = max(0, min(self.canvas_height, math.floor(y0f)))
        x1 = max(0, min(self.canvas_width, math.ceil(x1f)))
        y1 = max(0, min(self.canvas_height, math.ceil(y1f)))
        intersects_canvas = (
            x1f > 0 and y1f > 0 and x0f < self.canvas_width and y0f < self.canvas_height
        )
        if ensure_nonempty and intersects_canvas:
            x1 = min(self.canvas_width, max(x0 + 1, x1))
            y1 = min(self.canvas_height, max(y0 + 1, y1))
        return x0, y0, x1, y1


ObservationTransform = OverviewTransform | ROIObservationTransform


def make_observation_space(overview_size: int, viewport_size: int):
    from gymnasium import spaces

    return spaces.Dict(
        {
            "global": spaces.Box(0.0, 1.0, (5, overview_size, overview_size), dtype=np.float32),
            "viewport": spaces.Box(0.0, 1.0, (3, viewport_size, viewport_size), dtype=np.float32),
            "scalars": spaces.Box(0.0, 1.0, (5,), dtype=np.float32),
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


def crop_overview_to_roi(
    overview: np.ndarray,
    source_transform: OverviewTransform,
    roi_transform: ROIObservationTransform,
) -> NDArray[np.uint8]:
    """Crop a cached full-slide overview into the nominal snapped ROI.

    Resampling is performed directly from the existing overview, so this does
    not read or reinterpret any WSI patches.  The physical WSI intersection is
    copied onto a white canvas; snapped right/bottom overhang therefore remains
    white even when the ROI extends beyond the source dimensions.
    """

    source = np.asarray(overview)
    expected_shape = (
        source_transform.canvas_height,
        source_transform.canvas_width,
        3,
    )
    if source.shape != expected_shape:
        raise ValueError(
            f"Overview has shape {source.shape}; expected {expected_shape} for its transform"
        )
    if source.dtype != np.uint8:
        raise ValueError(f"Overview dtype is {source.dtype}; expected uint8")

    canvas = np.full(
        (roi_transform.canvas_height, roi_transform.canvas_width, 3),
        255,
        dtype=np.uint8,
    )
    source_box = source_transform.project_bounds(roi_transform.global_bounds)
    resized = Image.fromarray(source).transform(
        (roi_transform.content_width, roi_transform.content_height),
        Image.Transform.EXTENT,
        source_box,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(255, 255, 255),
    )
    resized_array = np.asarray(resized, dtype=np.uint8)

    roi_x0, roi_y0, roi_x1, roi_y1 = roi_transform.global_bounds
    valid_bounds = (
        max(roi_x0, 0),
        max(roi_y0, 0),
        min(roi_x1, source_transform.source_width),
        min(roi_y1, source_transform.source_height),
    )
    if valid_bounds[2] <= valid_bounds[0] or valid_bounds[3] <= valid_bounds[1]:
        return canvas

    x0, y0, x1, y1 = roi_transform.raster_bounds(valid_bounds, ensure_nonempty=False)
    source_x0 = max(0, x0 - roi_transform.content_x0)
    source_y0 = max(0, y0 - roi_transform.content_y0)
    source_x1 = min(roi_transform.content_width, x1 - roi_transform.content_x0)
    source_y1 = min(roi_transform.content_height, y1 - roi_transform.content_y0)
    if source_x1 > source_x0 and source_y1 > source_y0:
        destination_x0 = roi_transform.content_x0 + source_x0
        destination_y0 = roi_transform.content_y0 + source_y0
        destination_x1 = roi_transform.content_x0 + source_x1
        destination_y1 = roi_transform.content_y0 + source_y1
        canvas[destination_y0:destination_y1, destination_x0:destination_x1] = resized_array[
            source_y0:source_y1, source_x0:source_x1
        ]
    return canvas


def location_mask(transform: ObservationTransform, bounds: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((transform.canvas_height, transform.canvas_width), dtype=np.float32)
    x0, y0, x1, y1 = transform.raster_bounds(bounds)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def increment_visits(
    visits: np.ndarray,
    transform: ObservationTransform,
    bounds: tuple[int, int, int, int],
) -> None:
    expected_shape = (transform.canvas_height, transform.canvas_width)
    if visits.shape != expected_shape:
        raise ValueError(f"Visits have shape {visits.shape}; expected {expected_shape}")
    x0, y0, x1, y1 = transform.raster_bounds(bounds)
    region = visits[y0:y1, x0:x1]
    np.add(region, 1, out=region, casting="unsafe")


__all__ = [
    "ObservationTransform",
    "OverviewTransform",
    "ROIObservationTransform",
    "crop_overview_to_roi",
    "increment_visits",
    "location_mask",
    "make_observation_space",
    "rgb_chw_float",
]
