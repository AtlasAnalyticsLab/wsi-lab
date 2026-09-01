"""Reproducible training-trajectory visitation overlays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_cache import RasterTransform


def _transform(value: RasterTransform | Mapping[str, Any]) -> RasterTransform:
    return value if isinstance(value, RasterTransform) else RasterTransform.from_mapping(value)


def _point_bounds(point: Mapping[str, Any]) -> tuple[int, int, int, int]:
    raw = point.get("canonical_bounds", point.get("bounds"))
    if raw is None:
        raise ValueError("trajectory point has no canonical bounds")
    bounds = tuple(int(value) for value in raw)
    if len(bounds) != 4:
        raise ValueError("trajectory point bounds must contain four values")
    return bounds


def _is_view(point: Mapping[str, Any]) -> bool:
    explicit = point.get("is_view")
    if explicit is not None:
        return bool(explicit)
    action = point.get("action")
    return action is None or int(action) < 9


def visitation_density(
    transform: RasterTransform | Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
    *,
    equal_mass: bool = True,
) -> np.ndarray:
    """Rasterize view footprints; each view contributes unit mass by default."""

    resolved = _transform(transform)
    difference = np.zeros((resolved.canvas_height + 1, resolved.canvas_width + 1), dtype=np.float64)
    view_count = 0
    for point in trajectory:
        if not _is_view(point):
            continue
        x0, y0, x1, y1 = resolved.raster_bounds(_point_bounds(point))
        if x1 <= x0 or y1 <= y0:
            continue
        area = (x1 - x0) * (y1 - y0)
        weight = 1.0 / area if equal_mass else 1.0
        difference[y0, x0] += weight
        difference[y1, x0] -= weight
        difference[y0, x1] -= weight
        difference[y1, x1] += weight
        view_count += 1
    density = difference.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    if equal_mass and view_count and not np.isclose(float(density.sum()), view_count):
        raise AssertionError("equal-mass visitation raster lost view mass")
    return density.astype(np.float32)


def trajectory_centers(
    transform: RasterTransform | Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    resolved = _transform(transform)
    centers: list[tuple[float, float]] = []
    for point in trajectory:
        if not _is_view(point):
            continue
        x0, y0, x1, y1 = resolved.project_bounds(_point_bounds(point))
        centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def trajectory_magnifications(
    trajectory: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Magnification aligned one-to-one with :func:`trajectory_centers`."""

    values = [
        np.nan if point.get("magnification") is None else float(point["magnification"])
        for point in trajectory
        if _is_view(point)
    ]
    return np.asarray(values, dtype=np.float32)


def render_visit_overlay(
    overview: np.ndarray,
    transform: RasterTransform | Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    tumor_overlay: np.ndarray | None = None,
    title: str | None = None,
) -> Path:
    """Render density, path, start/end markers and an optional tumor outline."""

    resolved = _transform(transform)
    image = np.asarray(overview)
    if image.shape != (resolved.canvas_height, resolved.canvas_width, 3):
        raise ValueError("overview is not aligned with the supplied transform")
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if not trajectory:
        raise ValueError("trajectory cannot be empty")
    tumor: np.ndarray | None = None
    if tumor_overlay is not None:
        tumor = np.asarray(tumor_overlay, dtype=np.bool_)
        if tumor.shape != image.shape[:2]:
            raise ValueError("tumor overlay is not aligned with the overview")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    density = visitation_density(resolved, trajectory)
    centers = trajectory_centers(resolved, trajectory)
    magnifications = trajectory_magnifications(trajectory)
    positive = density[density > 0]
    heat = np.zeros_like(density)
    if len(positive):
        heat = np.log1p(density)
        maximum = float(heat.max())
        if maximum > 0:
            heat /= maximum

    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    axis.imshow(image)
    masked_heat = np.ma.masked_where(heat <= 0, heat)
    axis.imshow(masked_heat, cmap="magma", alpha=0.62, vmin=0.0, vmax=1.0)
    if tumor is not None and bool(tumor.any()):
        axis.contour(tumor.astype(np.uint8), levels=[0.5], colors=["#00ffff"], linewidths=1.2)
    if len(centers) > 1:
        axis.plot(centers[:, 0], centers[:, 1], color="white", linewidth=0.8, alpha=0.75)
    finite_magnifications = magnifications[np.isfinite(magnifications)]
    if len(finite_magnifications):
        unique_magnifications = sorted(float(value) for value in np.unique(finite_magnifications))
        color_map = plt.get_cmap("tab10")
        for index, magnification in enumerate(unique_magnifications):
            selected = np.isclose(magnifications, magnification)
            axis.scatter(
                centers[selected, 0],
                centers[selected, 1],
                color=color_map(index % 10),
                edgecolors="white",
                linewidths=0.45,
                s=18,
                marker="o",
                label=f"{magnification:g}x",
                zorder=4,
            )
        axis.legend(
            title="Magnification",
            loc="lower right",
            fontsize=7,
            title_fontsize=7,
            framealpha=0.8,
        )
    if len(centers):
        axis.scatter(centers[0, 0], centers[0, 1], c="#00ff66", s=34, marker="o", zorder=5)
        axis.scatter(centers[-1, 0], centers[-1, 1], c="#00b7ff", s=40, marker="X", zorder=5)
    axis.set_title(title or "Training-set visitation heatmap", fontsize=10)
    axis.axis("off")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=160,
        facecolor="white",
        metadata={"Software": "wsi-lab camelyon16 RL ablation"},
    )
    plt.close(figure)
    return destination


__all__ = [
    "render_visit_overlay",
    "trajectory_centers",
    "trajectory_magnifications",
    "visitation_density",
]
