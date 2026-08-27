from __future__ import annotations

from pathlib import Path

import numpy as np

from wsi_lab.envs.rendering import render_trajectory_figure


class RecordingTransform:
    def __init__(self) -> None:
        self.bounds: list[tuple[int, int, int, int]] = []

    def project_bounds(self, bounds):
        recorded = tuple(int(value) for value in bounds)
        self.bounds.append(recorded)
        return tuple(float(value) for value in recorded)


def test_trajectory_rendering_prefers_canonical_bounds_for_augmented_points(
    tmp_path: Path,
) -> None:
    transform = RecordingTransform()
    trajectory = [
        {
            "step": 0,
            "bounds": [30, 40, 50, 60],
            "canonical_bounds": [1, 2, 3, 4],
            "magnification": 2.5,
            "belief": 0.5,
        }
    ]

    destination = render_trajectory_figure(
        np.zeros((64, 64, 3), dtype=np.uint8),
        transform,  # type: ignore[arg-type]
        trajectory,
        tmp_path / "trajectory.png",
    )

    assert transform.bounds == [(1, 2, 3, 4)]
    assert destination.is_file()
