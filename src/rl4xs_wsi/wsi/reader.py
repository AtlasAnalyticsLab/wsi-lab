"""Exact level-0 region reading with deterministic white padding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from rl4xs_wsi.wsi.cache import PersistentViewportCache, ProcessLocalLRU, ViewportCache


class WSIBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SlideMetadata:
    path: Path
    dimensions: tuple[int, int]
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    mpp_x: float | None = None
    mpp_y: float | None = None
    objective_power: float | None = None
    vendor: str | None = None
    properties: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class OverviewTransform:
    """Aspect-preserving affine map from level-0 pixels to a white canvas."""

    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    content_x0: int
    content_y0: int
    content_width: int
    content_height: int

    @classmethod
    def fit(cls, source_size: tuple[int, int], canvas_size: tuple[int, int]) -> OverviewTransform:
        source_width, source_height = map(int, source_size)
        canvas_width, canvas_height = map(int, canvas_size)
        if min(source_width, source_height, canvas_width, canvas_height) <= 0:
            raise ValueError("source and canvas dimensions must be positive")
        scale = min(canvas_width / source_width, canvas_height / source_height)
        content_width = min(canvas_width, max(1, round(source_width * scale)))
        content_height = min(canvas_height, max(1, round(source_height * scale)))
        return cls(
            source_width=source_width,
            source_height=source_height,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            content_x0=(canvas_width - content_width) // 2,
            content_y0=(canvas_height - content_height) // 2,
            content_width=content_width,
            content_height=content_height,
        )

    @property
    def scale_x(self) -> float:
        return self.content_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.content_height / self.source_height

    def project_point(self, x: float, y: float) -> tuple[float, float]:
        return self.content_x0 + x * self.scale_x, self.content_y0 + y * self.scale_y

    def project_bounds(
        self, bounds: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x0, y0 = self.project_point(bounds[0], bounds[1])
        x1, y1 = self.project_point(bounds[2], bounds[3])
        return x0, y0, x1, y1

    def raster_bounds(
        self,
        bounds: tuple[float, float, float, float],
        *,
        ensure_nonempty: bool = True,
    ) -> tuple[int, int, int, int]:
        x0f, y0f, x1f, y1f = self.project_bounds(bounds)
        x0 = max(0, min(self.canvas_width, math.floor(x0f)))
        y0 = max(0, min(self.canvas_height, math.floor(y0f)))
        x1 = max(0, min(self.canvas_width, math.ceil(x1f)))
        y1 = max(0, min(self.canvas_height, math.ceil(y1f)))
        if ensure_nonempty and x0 < self.canvas_width and y0 < self.canvas_height:
            x1 = min(self.canvas_width, max(x0 + 1, x1))
            y1 = min(self.canvas_height, max(y0 + 1, y1))
        return x0, y0, x1, y1

    def to_dict(self) -> dict[str, int]:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "content_x0": self.content_x0,
            "content_y0": self.content_y0,
            "content_width": self.content_width,
            "content_height": self.content_height,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OverviewTransform:
        return cls(**{key: int(item) for key, item in value.items()})


class WSIReader(Protocol):
    def inspect(self, path: str | Path) -> SlideMetadata: ...

    def read_region(
        self,
        path: str | Path,
        bounds_level0: tuple[int, int, int, int],
        output_size: tuple[int, int],
    ) -> NDArray[np.uint8]: ...

    def read_overview(
        self, path: str | Path, canvas_size: tuple[int, int]
    ) -> tuple[NDArray[np.uint8], OverviewTransform]: ...

    def close(self) -> None: ...


def _property_float(properties: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        raw = properties.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _rgb_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


class OpenSlideReader:
    """OpenSlide backend whose handles are owned by the current worker process."""

    def __init__(
        self,
        *,
        handle_cache_size: int = 4,
        viewport_cache_size: int = 0,
        persistent_cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import openslide
        except (ImportError, OSError) as exc:
            raise WSIBackendUnavailable(
                "OpenSlide is unavailable; install the native OpenSlide library and openslide-python"
            ) from exc
        self._openslide = openslide
        self._handles: ProcessLocalLRU[Path, Any] = ProcessLocalLRU(
            handle_cache_size, on_evict=lambda handle: handle.close()
        )
        self._viewports = ViewportCache(viewport_cache_size)
        self._persistent_viewports = (
            PersistentViewportCache(persistent_cache_dir)
            if persistent_cache_dir is not None
            else None
        )

    def _handle(self, path: str | Path) -> Any:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        cached = self._handles.get(resolved)
        if cached is not None:
            return cached
        handle = self._openslide.OpenSlide(str(resolved))
        if self._handles.max_items == 0:
            return handle
        return self._handles.put(resolved, handle)

    def inspect(self, path: str | Path) -> SlideMetadata:
        resolved = Path(path).expanduser().resolve()
        handle = self._handle(resolved)
        properties = {str(key): str(value) for key, value in handle.properties.items()}
        metadata = SlideMetadata(
            path=resolved,
            dimensions=(int(handle.dimensions[0]), int(handle.dimensions[1])),
            level_dimensions=tuple(
                (int(value[0]), int(value[1])) for value in handle.level_dimensions
            ),
            level_downsamples=tuple(float(value) for value in handle.level_downsamples),
            mpp_x=_property_float(properties, "openslide.mpp-x"),
            mpp_y=_property_float(properties, "openslide.mpp-y"),
            objective_power=_property_float(
                properties, "openslide.objective-power", "aperio.AppMag", "hamamatsu.SourceLens"
            ),
            vendor=properties.get("openslide.vendor"),
            properties=properties,
        )
        if self._handles.max_items == 0:
            handle.close()
        return metadata

    def read_region(
        self,
        path: str | Path,
        bounds_level0: tuple[int, int, int, int],
        output_size: tuple[int, int],
    ) -> NDArray[np.uint8]:
        resolved = Path(path).expanduser().resolve()
        bounds = (
            int(bounds_level0[0]),
            int(bounds_level0[1]),
            int(bounds_level0[2]),
            int(bounds_level0[3]),
        )
        output = (int(output_size[0]), int(output_size[1]))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(f"invalid level-0 bounds: {bounds}")
        if min(output) <= 0:
            raise ValueError(f"invalid output size: {output}")
        cache_key = (str(resolved), bounds, output)
        cached = self._viewports.get(cache_key)
        if cached is not None:
            return cached
        if self._persistent_viewports is not None:
            cached = self._persistent_viewports.get(resolved, bounds, output)
            if cached is not None:
                self._viewports.put(cache_key, cached)
                return cached
        handle = self._handle(resolved)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        desired_downsample = max(width / output[0], height / output[1], 1.0)
        level = int(handle.get_best_level_for_downsample(desired_downsample))
        downsample = float(handle.level_downsamples[level])
        read_size = (max(1, math.ceil(width / downsample)), max(1, math.ceil(height / downsample)))
        region = _rgb_on_white(handle.read_region((bounds[0], bounds[1]), level, read_size))
        if region.size != output:
            region = region.resize(output, Image.Resampling.BILINEAR)
        array = np.asarray(region, dtype=np.uint8)
        if self._persistent_viewports is not None:
            self._persistent_viewports.put(resolved, bounds, output, array)
        self._viewports.put(cache_key, array)
        if self._handles.max_items == 0:
            handle.close()
        return array.copy()

    def read_overview(
        self, path: str | Path, canvas_size: tuple[int, int]
    ) -> tuple[NDArray[np.uint8], OverviewTransform]:
        resolved = Path(path).expanduser().resolve()
        canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        handle = self._handle(resolved)
        transform = OverviewTransform.fit(
            (int(handle.dimensions[0]), int(handle.dimensions[1])), canvas_size
        )
        thumbnail = _rgb_on_white(
            handle.get_thumbnail((transform.content_width, transform.content_height))
        )
        if thumbnail.size != (transform.content_width, transform.content_height):
            thumbnail = thumbnail.resize(
                (transform.content_width, transform.content_height), Image.Resampling.BILINEAR
            )
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))
        canvas.paste(thumbnail, (transform.content_x0, transform.content_y0))
        if self._handles.max_items == 0:
            handle.close()
        return np.asarray(canvas, dtype=np.uint8), transform

    def close(self) -> None:
        self._viewports.clear()
        self._handles.clear()

    def __enter__(self) -> OpenSlideReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PillowSlideReader:
    """Flat-image backend for synthetic fixtures and small non-pyramidal slides."""

    def inspect(self, path: str | Path) -> SlideMetadata:
        resolved = Path(path).expanduser().resolve()
        with Image.open(resolved) as image:
            dimensions = (int(image.size[0]), int(image.size[1]))
        return SlideMetadata(
            path=resolved,
            dimensions=dimensions,
            level_dimensions=(dimensions,),
            level_downsamples=(1.0,),
        )

    def read_region(
        self,
        path: str | Path,
        bounds_level0: tuple[int, int, int, int],
        output_size: tuple[int, int],
    ) -> NDArray[np.uint8]:
        x0, y0, x1, y1 = map(int, bounds_level0)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid level-0 bounds: {bounds_level0}")
        with Image.open(Path(path)) as source_handle:
            source = source_handle.convert("RGB")
            canvas = Image.new("RGB", (x1 - x0, y1 - y0), (255, 255, 255))
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(source.width, x1), min(source.height, y1)
            if sx1 > sx0 and sy1 > sy0:
                crop = source.crop((sx0, sy0, sx1, sy1))
                canvas.paste(crop, (sx0 - x0, sy0 - y0))
            canvas = canvas.resize(
                (int(output_size[0]), int(output_size[1])), Image.Resampling.BILINEAR
            )
            return np.asarray(canvas, dtype=np.uint8)

    def read_overview(
        self, path: str | Path, canvas_size: tuple[int, int]
    ) -> tuple[NDArray[np.uint8], OverviewTransform]:
        with Image.open(Path(path)) as source_handle:
            source = source_handle.convert("RGB")
            requested_canvas = (int(canvas_size[0]), int(canvas_size[1]))
            transform = OverviewTransform.fit(
                (int(source.size[0]), int(source.size[1])), requested_canvas
            )
            resized = source.resize(
                (transform.content_width, transform.content_height), Image.Resampling.BILINEAR
            )
            canvas = Image.new("RGB", requested_canvas, (255, 255, 255))
            canvas.paste(resized, (transform.content_x0, transform.content_y0))
            return np.asarray(canvas, dtype=np.uint8), transform

    def close(self) -> None:
        return None


__all__ = [
    "OpenSlideReader",
    "OverviewTransform",
    "PillowSlideReader",
    "SlideMetadata",
    "WSIBackendUnavailable",
    "WSIReader",
]
