"""Dependency-light trajectory/state rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wsi_lab.envs.observation import ObservationTransform, OverviewTransform


def render_state(
    overview: np.ndarray,
    transform: ObservationTransform,
    bounds: tuple[int, int, int, int],
    visits: np.ndarray,
    max_visit_count: int,
) -> np.ndarray:
    image = np.asarray(overview).copy()
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    heat = np.minimum(visits, max_visit_count).astype(np.float32) / max_visit_count
    image[..., 0] = np.maximum(image[..., 0], (heat * 180).astype(np.uint8))
    x0, y0, x1, y1 = transform.raster_bounds(bounds)
    image[y0:y1, x0] = (0, 255, 0)
    image[y0:y1, x1 - 1] = (0, 255, 0)
    image[y0, x0:x1] = (0, 255, 0)
    image[y1 - 1, x0:x1] = (0, 255, 0)
    return image


def render_trajectory_figure(
    overview: np.ndarray,
    transform: OverviewTransform,
    trajectory: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    current_viewport: np.ndarray | None = None,
    tumor_overlay: np.ndarray | None = None,
) -> Path:
    """Save an analysis figure without requiring tumor annotations."""
    if not trajectory:
        raise ValueError("trajectory cannot be empty")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    image = np.asarray(overview)
    axes[0].imshow(image)
    if tumor_overlay is not None:
        overlay = np.ma.masked_where(~np.asarray(tumor_overlay, dtype=np.bool_), tumor_overlay)
        axes[0].imshow(overlay, cmap="Reds", alpha=0.25, interpolation="nearest")
    centers: list[tuple[float, float]] = []
    magnifications: list[float] = []
    beliefs: list[float] = []
    steps: list[int] = []
    for point in trajectory:
        # Runtime-augmented trajectories expose agent-facing ``bounds`` while
        # retaining canonical bounds for plots drawn over persisted overviews.
        bounds_value = point["canonical_bounds"] if "canonical_bounds" in point else point["bounds"]
        raw_bounds = tuple(int(value) for value in bounds_value)
        if len(raw_bounds) != 4:
            raise ValueError(f"trajectory bounds must have four values, got {raw_bounds}")
        bounds = (raw_bounds[0], raw_bounds[1], raw_bounds[2], raw_bounds[3])
        x0, y0, x1, y1 = transform.project_bounds(bounds)
        centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
        magnifications.append(float(point["magnification"]))
        beliefs.append(float(point["belief"]))
        steps.append(int(point["step"]))
    if len(centers) > 1:
        segments = np.stack((np.asarray(centers[:-1]), np.asarray(centers[1:])), axis=1)
        collection = LineCollection(segments.tolist(), cmap="viridis", linewidths=2)
        collection.set_array(np.asarray(magnifications[1:], dtype=np.float32))
        axes[0].add_collection(collection)
    axes[0].scatter(*np.asarray(centers).T, c=magnifications, cmap="viridis", s=14)
    axes[0].set_title("Navigation path and magnification")
    axes[0].axis("off")
    axes[1].plot(steps, beliefs, color="tab:blue")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Action", ylabel="P(cancer)", ylim=(0, 1), title="Belief trajectory")
    if current_viewport is None:
        axes[2].text(0.5, 0.5, "Viewport not recorded", ha="center", va="center")
    else:
        axes[2].imshow(current_viewport)
    axes[2].set_title("Final viewport")
    axes[2].axis("off")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


__all__ = ["render_state", "render_trajectory_figure"]
