"""Matplotlib dashboard for interactive manual WSI navigation."""

from __future__ import annotations

from typing import Any

import numpy as np

from wsi_lab.manual.controller import ACTION_KEY_BINDINGS, ManualController

SCALAR_NAMES = (
    "magnification",
    "remaining budget",
    "ABMIL belief",
    "tissue-ROI center x",
    "tissue-ROI center y",
)


def observation_panels(
    observation: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert actor observations into display arrays without reading hidden state."""

    viewport = np.asarray(observation["viewport"], dtype=np.float32)
    global_observation = np.asarray(observation["global"], dtype=np.float32)
    if viewport.ndim != 3 or viewport.shape[0] != 3:
        raise ValueError(f"viewport observation must be CHW RGB, got {viewport.shape}")
    if global_observation.ndim != 3 or global_observation.shape[0] != 5:
        raise ValueError(
            "global observation must contain RGB, location, and visit channels; "
            f"got {global_observation.shape}"
        )
    viewport_rgb = np.clip(viewport.transpose(1, 2, 0), 0.0, 1.0)
    overview_rgb = np.clip(global_observation[:3].transpose(1, 2, 0), 0.0, 1.0)
    location = np.clip(global_observation[3], 0.0, 1.0)
    visits = np.clip(global_observation[4], 0.0, 1.0)

    # The two raw mask panels remain visible; this composite is only a spatial aid.
    location_alpha = 0.55 * location[..., None]
    visit_alpha = 0.45 * visits[..., None] * (1.0 - location_alpha)
    composite = overview_rgb * (1.0 - location_alpha - visit_alpha)
    composite += location_alpha * np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
    composite += visit_alpha * np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    return viewport_rgb, overview_rgb, np.clip(composite, 0.0, 1.0), location, visits


class ManualPlayWindow:
    """Interactive dashboard bound to one :class:`ManualController`."""

    def __init__(self, controller: ManualController) -> None:
        import matplotlib.pyplot as plt

        if controller.observation is None or controller.info is None:
            raise RuntimeError("manual controller must be reset before creating the window")
        self.controller = controller
        self._plt = plt
        self.figure = plt.figure(figsize=(19, 9.5))
        grid = self.figure.add_gridspec(2, 10, height_ratios=(2.1, 1.25))
        self.viewport_axis = self.figure.add_subplot(grid[0, 0:2])
        self.overview_axis = self.figure.add_subplot(grid[0, 2:4])
        self.composite_axis = self.figure.add_subplot(grid[0, 4:6])
        self.location_axis = self.figure.add_subplot(grid[0, 6:8])
        self.visits_axis = self.figure.add_subplot(grid[0, 8:10])
        self.status_axis = self.figure.add_subplot(grid[1, 0:5])
        self.actions_axis = self.figure.add_subplot(grid[1, 5:10])
        self._connection_id = self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("WSI Lab manual controller")
        self.refresh()

    def _status_text(self) -> str:
        controller = self.controller
        assert controller.observation is not None and controller.info is not None
        info = controller.info
        scalars = np.asarray(controller.observation["scalars"], dtype=np.float32).reshape(-1)
        scalar_lines = []
        for index, value in enumerate(scalars):
            name = SCALAR_NAMES[index] if index < len(SCALAR_NAMES) else f"scalar {index}"
            scalar_lines.append(f"  {name:<21} {float(value):.6f}")
        status = (
            "terminated"
            if controller.terminated
            else "truncated"
            if controller.truncated
            else "active"
        )
        latest_action = (
            "none" if controller.latest_action is None else controller.latest_action.name
        )
        latest_reward = (
            "none" if controller.latest_reward is None else f"{controller.latest_reward:+.6f}"
        )
        outcome = ""
        if controller.terminated and "correct" in info:
            outcome = f"\nDiagnosis correct: {bool(info['correct'])}"
        return (
            f"Slide: {info.get('slide_id')}\n"
            f"Cell: {info.get('cell_id')}  row/column: {info.get('row')}/{info.get('column')}\n"
            f"Episode: {status}  step: {info.get('actions_taken')}  "
            f"remaining: {info.get('remaining_actions')}\n"
            f"Cumulative reward: {controller.cumulative_reward:+.6f}\n"
            f"Latest action: {latest_action}  latest reward: {latest_reward}\n"
            f"Message: {controller.latest_message}\n"
            f"Evidence: {info.get('evidence_count')}  magnification: {info.get('magnification'):g}x  "
            f"belief: {float(info.get('belief', 0.0)):.6f}\n"
            "Actor scalar observation (normalized; x/y use snapped tissue ROI):\n"
            + "\n".join(scalar_lines)
            + outcome
        )

    def _actions_text(self) -> str:
        mask = self.controller.action_mask
        lines = ["Keyboard actions (validity comes from info['action_mask']):"]
        for binding in ACTION_KEY_BINDINGS:
            valid = bool(mask[int(binding.action)])
            marker = "VALID  " if valid else "INVALID"
            lines.append(
                f"  {binding.key:>9}  {marker}  {binding.action.value:>2}: "
                f"{binding.action.name} ({binding.label})"
            )
        lines.extend(
            (
                "",
                "Episode controls:",
                "  r reset current slide     n or ] next slide     [ previous slide",
                "  Esc close window          Backspace also zooms out",
            )
        )
        return "\n".join(lines)

    def refresh(self) -> None:
        observation = self.controller.observation
        info = self.controller.info
        if observation is None or info is None:
            raise RuntimeError("manual controller is not initialized")
        viewport, overview, composite, location, visits = observation_panels(observation)
        image_panels: tuple[tuple[Any, np.ndarray, str, dict[str, Any]], ...] = (
            (self.viewport_axis, viewport, "Current viewport (global WSI cell)", {}),
            (self.overview_axis, overview, "Tissue ROI overview RGB (global 0-2)", {}),
            (
                self.composite_axis,
                composite,
                "Tissue ROI overview + location/visits",
                {},
            ),
            (
                self.location_axis,
                location,
                "Tissue ROI current location (global 3)",
                {"cmap": "Blues"},
            ),
            (
                self.visits_axis,
                visits,
                "Tissue ROI visited map (global 4)",
                {"cmap": "magma"},
            ),
        )
        for axis, image, title, kwargs in image_panels:
            axis.clear()
            axis.imshow(image, vmin=0.0, vmax=1.0, interpolation="nearest", **kwargs)
            axis.set_title(title)
            axis.set_axis_off()
        self.status_axis.clear()
        self.status_axis.set_axis_off()
        self.status_axis.text(
            0.0,
            1.0,
            self._status_text(),
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )
        self.actions_axis.clear()
        self.actions_axis.set_axis_off()
        self.actions_axis.text(
            0.0,
            1.0,
            self._actions_text(),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )
        self.figure.suptitle(
            f"WSI Lab manual play — {info.get('slide_id')} — {info.get('magnification'):g}x",
            fontsize=14,
        )
        self.figure.canvas.draw_idle()

    def _on_key(self, event: Any) -> None:
        result = self.controller.handle_key(getattr(event, "key", None))
        if result.close_requested:
            self._plt.close(self.figure)
            return
        self.refresh()

    def show(self) -> None:
        self._plt.show()


def run_manual_play(
    environment: Any,
    *,
    slide_ids: tuple[str, ...],
    initial_slide_id: str | None,
    seed: int | None,
) -> None:
    """Reset a production environment and run its interactive dashboard."""

    import matplotlib.pyplot as plt

    backend = str(plt.get_backend()).lower()
    if backend in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
        raise RuntimeError(
            f"manual-play requires an interactive Matplotlib backend; current backend is {backend!r}"
        )
    controller = ManualController(environment, slide_ids=slide_ids, seed=seed)
    controller.reset(initial_slide_id)
    window = ManualPlayWindow(controller)
    window.show()


__all__ = ["SCALAR_NAMES", "ManualPlayWindow", "observation_panels", "run_manual_play"]
