#!/usr/bin/env python3
"""Manually navigate CAMELYON16 test_001 and save its final spatial observations.

The window uses the production RL environment and the same keyboard controls as
``wsi-lab manual-play``. Only navigation actions are accepted. After 20 valid
environment steps, the actor-facing visit channel and current-location channel
are saved as separate PNG figures and the window closes automatically.

Run from the repository root in the ``wsi-lab`` environment::

    conda run --no-capture-output -n wsi-lab \
      python scripts/capture_manual_observations.py --allow-test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from wsi_lab import cli
from wsi_lab.config import load_config
from wsi_lab.data.factory import create_dataset_adapter
from wsi_lab.envs.actions import NAVIGATION_ACTIONS
from wsi_lab.manual.controller import ACTION_KEY_BINDINGS, ManualController
from wsi_lab.manual.ui import ManualPlayWindow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "camelyon16.yaml"
CAMELYON16_ROOT = Path("/home/ahmed/SharedData/Datasets/CAMELYON16")
SLIDE_PATH = CAMELYON16_ROOT / "images" / "test_001.tif"
SLIDE_ID = "test_001"
DEFAULT_OUTPUT_DIR = Path("/home/ahmed/SharedData/Datasets/temp_data")
STEP_LIMIT = 20

_NAVIGATION_KEYS = {
    binding.key
    for binding in ACTION_KEY_BINDINGS
    if binding.action in NAVIGATION_ACTIONS
}
_NAVIGATION_KEYS.add("backspace")
_DISABLED_EPISODE_KEYS = {"r", "n", "]", "["}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manually navigate CAMELYON16 test_001 for 20 valid steps, then save "
            "the actor-facing visit heatmap and location mask."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required explicit authorization to open the official CAMELYON16 test slide.",
    )
    return parser


def _save_mask_figure(
    values: np.ndarray,
    destination: Path,
    *,
    title: str,
    cmap: str,
    colorbar_label: str,
) -> Path:
    import matplotlib.pyplot as plt

    image = np.asarray(values, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"spatial observation channel must be two-dimensional, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("spatial observation channel contains non-finite values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    artist = axis.imshow(
        image,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("Tissue-ROI observation column")
    axis.set_ylabel("Tissue-ROI observation row")
    colorbar = figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)

    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=200)
        temporary.replace(destination)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return destination


def save_final_observation(
    observation: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save the exact actor-facing global channels 4 and 3, respectively."""

    global_observation = np.asarray(observation["global"], dtype=np.float32)
    if global_observation.ndim != 3 or global_observation.shape[0] != 5:
        raise ValueError(
            "global observation must contain RGB, location, and visit channels; "
            f"got {global_observation.shape}"
        )

    destination = output_dir.expanduser().resolve()
    visit_path = _save_mask_figure(
        global_observation[4],
        destination / f"{SLIDE_ID}_visit_heatmap.png",
        title=f"{SLIDE_ID}: actor visit heatmap after {STEP_LIMIT} steps",
        cmap="magma",
        colorbar_label="Scaled visit count",
    )
    location_path = _save_mask_figure(
        global_observation[3],
        destination / f"{SLIDE_ID}_location_mask.png",
        title=f"{SLIDE_ID}: actor location mask after {STEP_LIMIT} steps",
        cmap="Blues",
        colorbar_label="Current viewport mask",
    )
    return visit_path, location_path


class ObservationCaptureWindow(ManualPlayWindow):
    """Manual-play window that stops and saves after a fixed navigation budget."""

    def __init__(self, controller: ManualController, *, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.saved_paths: tuple[Path, Path] | None = None
        super().__init__(controller)

    def _status_text(self) -> str:
        return (
            super()._status_text()
            + f"\n\nCapture progress: {self._actions_taken()}/{STEP_LIMIT} valid navigation steps"
            + "\nReset, slide selection, and diagnosis are disabled for this capture."
        )

    def _actions_taken(self) -> int:
        info = self.controller.info
        return 0 if info is None else int(info.get("actions_taken", 0))

    def _block(self, message: str) -> None:
        self.controller.latest_action_blocked = True
        self.controller.latest_message = message
        self.refresh()

    def _on_key(self, event: Any) -> None:
        key = "" if getattr(event, "key", None) is None else str(event.key).lower()
        if key == "escape":
            self._plt.close(self.figure)
            return
        if self.saved_paths is not None:
            return
        if key in _DISABLED_EPISODE_KEYS:
            self._block("reset and slide selection are disabled during the 20-step capture")
            return
        if key not in _NAVIGATION_KEYS:
            self._block(f"only viewport navigation keys are enabled; ignored {key!r}")
            return

        result = self.controller.handle_key(key)
        self.refresh()
        if not result.state_changed or self._actions_taken() < STEP_LIMIT:
            return

        observation = self.controller.observation
        if observation is None:
            raise RuntimeError("manual controller has no final observation to save")
        self.saved_paths = save_final_observation(observation, self.output_dir)
        print(f"Saved visit heatmap: {self.saved_paths[0]}", flush=True)
        print(f"Saved location mask: {self.saved_paths[1]}", flush=True)
        self._plt.close(self.figure)


def _require_interactive_backend() -> None:
    import matplotlib.pyplot as plt

    backend = str(plt.get_backend()).lower()
    if backend in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
        raise RuntimeError(
            "manual capture requires an interactive Matplotlib backend; "
            f"current backend is {backend!r}. Use a desktop session or X forwarding."
        )


def _validate_fixed_slide(config: Any) -> None:
    if not SLIDE_PATH.is_file():
        raise FileNotFoundError(f"CAMELYON16 slide does not exist: {SLIDE_PATH}")
    adapter = create_dataset_adapter(config.dataset, config.embeddings)
    configured_path = adapter.get_slide(SLIDE_ID).slide_path.resolve()
    if configured_path != SLIDE_PATH.resolve():
        raise ValueError(
            f"configuration resolves {SLIDE_ID} to {configured_path}, not the required {SLIDE_PATH}"
        )
    split = cli._load_verified_split(config)
    if SLIDE_ID not in split.ids("test"):
        raise ValueError(f"{SLIDE_ID} is not in the persisted official test split")
    if int(config.environment.max_episode_steps) <= STEP_LIMIT:
        raise ValueError(
            "environment.max_episode_steps must exceed 20 so all 20 capture steps "
            "remain navigation actions"
        )


def main() -> int:
    args = _parser().parse_args()
    if not args.allow_test:
        raise RuntimeError("Official test access requires the explicit --allow-test flag")

    os.environ.setdefault("CAMELYON16_ROOT", str(CAMELYON16_ROOT))
    config = load_config(args.config)
    _validate_fixed_slide(config)
    _require_interactive_backend()

    # This is the guarded route also used by official RL test evaluation. It is
    # idempotent and reuses compatible navigation artifacts when they exist.
    cli._prepare_authorized_test_preprocessing(config)
    environment = cli._make_production_environment(
        config,
        split_name="test",
        evaluation=True,
        allow_test=True,
    )
    controller = ManualController(
        environment,
        slide_ids=(SLIDE_ID,),
        seed=config.rl.seed if args.seed is None else args.seed,
    )
    try:
        controller.reset(SLIDE_ID)
        window = ObservationCaptureWindow(controller, output_dir=args.output_dir)
        window.show()
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
