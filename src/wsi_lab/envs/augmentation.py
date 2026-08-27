"""Episode-consistent geometric and H&E augmentations for WSI observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wsi_lab.envs.actions import PAN_ACTIONS, ZOOM_IN_ACTIONS, Action

_PAN_VECTORS = {
    Action.PAN_UP: (0, -1),
    Action.PAN_DOWN: (0, 1),
    Action.PAN_LEFT: (-1, 0),
    Action.PAN_RIGHT: (1, 0),
}
_VECTOR_TO_PAN = {vector: action for action, vector in _PAN_VECTORS.items()}
_ZOOM_VECTORS = {
    Action.ZOOM_IN_TL: (-1, -1),
    Action.ZOOM_IN_TR: (1, -1),
    Action.ZOOM_IN_BL: (-1, 1),
    Action.ZOOM_IN_BR: (1, 1),
}
_VECTOR_TO_ZOOM = {vector: action for action, vector in _ZOOM_VECTORS.items()}


def _positive_dimensions(first: int, second: int, *, label: str) -> tuple[int, int]:
    values = (int(first), int(second))
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"{label} must be positive, got {values}")
    return values


@dataclass(frozen=True, slots=True)
class GeometricOrientation:
    """One of eight square symmetries applied to a zero-origin padded frame.

    ``quarter_turns`` counts clockwise rotations. The optional horizontal flip
    is applied in the rotated/displayed frame. Point coordinates describe pixel
    edges, so a canonical frame spans ``[0, width] x [0, height]``.
    """

    quarter_turns: int = 0
    horizontal_flip: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.quarter_turns, bool) or not isinstance(self.quarter_turns, int):
            raise TypeError("quarter_turns must be an integer")
        if not 0 <= self.quarter_turns <= 3:
            raise ValueError("quarter_turns must be in [0, 3]")
        if not isinstance(self.horizontal_flip, bool):
            raise TypeError("horizontal_flip must be a boolean")

    @classmethod
    def sample(cls, rng: np.random.Generator) -> GeometricOrientation:
        """Sample uniformly from the eight supported orientations."""

        index = int(rng.integers(0, 8))
        return cls(quarter_turns=index % 4, horizontal_flip=index >= 4)

    @property
    def name(self) -> str:
        suffix = "_hflip" if self.horizontal_flip else ""
        return f"rot{self.quarter_turns * 90}{suffix}"

    def transformed_size(self, width: int, height: int) -> tuple[int, int]:
        """Return displayed ``(width, height)`` for a canonical frame."""

        width, height = _positive_dimensions(width, height, label="frame dimensions")
        if self.quarter_turns % 2:
            return height, width
        return width, height

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Apply this orientation to an image or spatial channel array."""

        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError(f"image must have at least two dimensions, got {array.shape}")
        transformed = np.rot90(array, k=-self.quarter_turns, axes=(0, 1))
        if self.horizontal_flip:
            transformed = np.flip(transformed, axis=1)
        return np.ascontiguousarray(transformed)

    def inverse_image(self, image: np.ndarray) -> np.ndarray:
        """Map an oriented image back to its canonical orientation."""

        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError(f"image must have at least two dimensions, got {array.shape}")
        if self.horizontal_flip:
            array = np.flip(array, axis=1)
        restored = np.rot90(array, k=self.quarter_turns, axes=(0, 1))
        return np.ascontiguousarray(restored)

    def transform_point(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        """Transform a point from canonical into displayed edge coordinates."""

        width, height = _positive_dimensions(width, height, label="frame dimensions")
        x_value = float(x)
        y_value = float(y)
        if self.quarter_turns == 0:
            transformed_x, transformed_y = x_value, y_value
        elif self.quarter_turns == 1:
            transformed_x, transformed_y = height - y_value, x_value
        elif self.quarter_turns == 2:
            transformed_x, transformed_y = width - x_value, height - y_value
        else:
            transformed_x, transformed_y = y_value, width - x_value
        transformed_width, _ = self.transformed_size(width, height)
        if self.horizontal_flip:
            transformed_x = transformed_width - transformed_x
        return transformed_x, transformed_y

    def inverse_point(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        """Map a displayed point back into canonical edge coordinates.

        ``width`` and ``height`` are always the canonical frame dimensions.
        """

        width, height = _positive_dimensions(width, height, label="frame dimensions")
        transformed_width, _ = self.transformed_size(width, height)
        transformed_x = float(x)
        transformed_y = float(y)
        if self.horizontal_flip:
            transformed_x = transformed_width - transformed_x
        if self.quarter_turns == 0:
            return transformed_x, transformed_y
        if self.quarter_turns == 1:
            return transformed_y, height - transformed_x
        if self.quarter_turns == 2:
            return width - transformed_x, height - transformed_y
        return width - transformed_y, transformed_x

    def transform_bounds(
        self,
        bounds: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        """Transform canonical half-open bounds into displayed coordinates."""

        x0, y0, x1, y1 = bounds
        if x1 < x0 or y1 < y0:
            raise ValueError(f"bounds must be ordered, got {bounds}")
        corners = (
            self.transform_point(x0, y0, width, height),
            self.transform_point(x1, y0, width, height),
            self.transform_point(x0, y1, width, height),
            self.transform_point(x1, y1, width, height),
        )
        xs, ys = zip(*corners, strict=True)
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def inverse_bounds(
        self,
        bounds: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        """Map displayed half-open bounds back to canonical coordinates."""

        x0, y0, x1, y1 = bounds
        if x1 < x0 or y1 < y0:
            raise ValueError(f"bounds must be ordered, got {bounds}")
        corners = (
            self.inverse_point(x0, y0, width, height),
            self.inverse_point(x1, y0, width, height),
            self.inverse_point(x0, y1, width, height),
            self.inverse_point(x1, y1, width, height),
        )
        xs, ys = zip(*corners, strict=True)
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def transformed_grid_shape(self, rows: int, columns: int) -> tuple[int, int]:
        """Return displayed ``(rows, columns)`` for a canonical grid."""

        rows, columns = _positive_dimensions(rows, columns, label="grid dimensions")
        if self.quarter_turns % 2:
            return columns, rows
        return rows, columns

    def transform_row_column(
        self,
        row: int,
        column: int,
        rows: int,
        columns: int,
    ) -> tuple[int, int]:
        """Transform a canonical grid index into its displayed row and column."""

        rows, columns = _positive_dimensions(rows, columns, label="grid dimensions")
        if not 0 <= row < rows or not 0 <= column < columns:
            raise IndexError((row, column))
        if self.quarter_turns == 0:
            transformed_row, transformed_column = row, column
        elif self.quarter_turns == 1:
            transformed_row, transformed_column = column, rows - 1 - row
        elif self.quarter_turns == 2:
            transformed_row, transformed_column = rows - 1 - row, columns - 1 - column
        else:
            transformed_row, transformed_column = columns - 1 - column, row
        _, transformed_columns = self.transformed_grid_shape(rows, columns)
        if self.horizontal_flip:
            transformed_column = transformed_columns - 1 - transformed_column
        return transformed_row, transformed_column

    def inverse_row_column(
        self,
        row: int,
        column: int,
        rows: int,
        columns: int,
    ) -> tuple[int, int]:
        """Map a displayed grid index back to canonical row and column."""

        rows, columns = _positive_dimensions(rows, columns, label="grid dimensions")
        transformed_rows, transformed_columns = self.transformed_grid_shape(rows, columns)
        if not 0 <= row < transformed_rows or not 0 <= column < transformed_columns:
            raise IndexError((row, column))
        transformed_column = column
        if self.horizontal_flip:
            transformed_column = transformed_columns - 1 - transformed_column
        if self.quarter_turns == 0:
            return row, transformed_column
        if self.quarter_turns == 1:
            return rows - 1 - transformed_column, row
        if self.quarter_turns == 2:
            return rows - 1 - row, columns - 1 - transformed_column
        return transformed_column, columns - 1 - row

    def _transform_vector(self, vector: tuple[int, int]) -> tuple[int, int]:
        x_value, y_value = vector
        if self.quarter_turns == 0:
            transformed = (x_value, y_value)
        elif self.quarter_turns == 1:
            transformed = (-y_value, x_value)
        elif self.quarter_turns == 2:
            transformed = (-x_value, -y_value)
        else:
            transformed = (y_value, -x_value)
        if self.horizontal_flip:
            return -transformed[0], transformed[1]
        return transformed

    def from_canonical_action(self, action: Action | int) -> Action:
        """Map a canonical action into the equivalent agent-facing action."""

        canonical = Action(int(action))
        if canonical in PAN_ACTIONS:
            return _VECTOR_TO_PAN[self._transform_vector(_PAN_VECTORS[canonical])]
        if canonical in ZOOM_IN_ACTIONS:
            return _VECTOR_TO_ZOOM[self._transform_vector(_ZOOM_VECTORS[canonical])]
        return canonical

    def to_canonical_action(self, action: Action | int) -> Action:
        """Map an agent-facing action to the canonical navigation graph action."""

        displayed = Action(int(action))
        if displayed not in PAN_ACTIONS and displayed not in ZOOM_IN_ACTIONS:
            return displayed
        candidates = PAN_ACTIONS if displayed in PAN_ACTIONS else ZOOM_IN_ACTIONS
        return next(
            canonical
            for canonical in candidates
            if self.from_canonical_action(canonical) is displayed
        )

    def mask_from_canonical(self, canonical_mask: np.ndarray) -> np.ndarray:
        """Permute a canonical action mask into agent-facing action order."""

        mask = np.asarray(canonical_mask)
        if mask.ndim != 1 or mask.shape[0] != len(Action):
            raise ValueError(f"action mask must have shape ({len(Action)},), got {mask.shape}")
        displayed = np.empty_like(mask)
        for agent_action in Action:
            displayed[agent_action] = mask[self.to_canonical_action(agent_action)]
        return np.ascontiguousarray(displayed)


# Columns are normalized optical-density vectors for hematoxylin, eosin, and
# their orthogonal residual. The residual coefficient is deliberately left
# unchanged by HEStainAugmentation.
_HEMATOXYLIN_OD = np.asarray((0.650, 0.704, 0.286), dtype=np.float64)
_EOSIN_OD = np.asarray((0.072, 0.990, 0.105), dtype=np.float64)
_HEMATOXYLIN_OD /= np.linalg.norm(_HEMATOXYLIN_OD)
_EOSIN_OD /= np.linalg.norm(_EOSIN_OD)
_RESIDUAL_OD = np.cross(_HEMATOXYLIN_OD, _EOSIN_OD)
_RESIDUAL_OD /= np.linalg.norm(_RESIDUAL_OD)
_STAIN_BASIS = np.column_stack((_HEMATOXYLIN_OD, _EOSIN_OD, _RESIDUAL_OD))
_INVERSE_STAIN_BASIS = np.linalg.inv(_STAIN_BASIS)


@dataclass(frozen=True, slots=True)
class HEStainAugmentation:
    """Fixed-basis optical-density H&E concentration augmentation."""

    hematoxylin_scale: float = 1.0
    eosin_scale: float = 1.0

    def __post_init__(self) -> None:
        scales = (float(self.hematoxylin_scale), float(self.eosin_scale))
        if not all(np.isfinite(scale) and scale > 0.0 for scale in scales):
            raise ValueError(f"stain scales must be finite and positive, got {scales}")

    @classmethod
    def sample(
        cls,
        rng: np.random.Generator,
        *,
        low: float = 0.9,
        high: float = 1.1,
    ) -> HEStainAugmentation:
        """Independently sample H and E concentration scales."""

        if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high < low:
            raise ValueError(f"invalid stain scale interval [{low}, {high}]")
        return cls(
            hematoxylin_scale=float(rng.uniform(low, high)),
            eosin_scale=float(rng.uniform(low, high)),
        )

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Apply the sampled stain scales and return contiguous uint8 RGB."""

        rgb = np.asarray(image)
        if rgb.ndim < 3 or rgb.shape[-1] != 3:
            raise ValueError(f"image must have shape (..., 3), got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise TypeError(f"image must have dtype uint8, got {rgb.dtype}")
        if self.hematoxylin_scale == 1.0 and self.eosin_scale == 1.0:
            return np.ascontiguousarray(rgb.copy())

        original_shape = rgb.shape
        flat_rgb = rgb.reshape(-1, 3).astype(np.float64)
        white = np.all(flat_rgb == 255.0, axis=1)
        optical_density = -np.log((flat_rgb + 1.0) / 256.0)
        # Collapse deconvolution, concentration scaling, and reconstruction
        # into one 3x3 transform. This is algebraically identical to two
        # per-pixel matrix products but materially cheaper for every viewport.
        od_transform = (
            _INVERSE_STAIN_BASIS.T
            @ np.diag((self.hematoxylin_scale, self.eosin_scale, 1.0))
            @ _STAIN_BASIS.T
        )
        augmented_od = optical_density @ od_transform
        augmented = np.rint(256.0 * np.exp(-augmented_od) - 1.0)
        augmented = np.clip(augmented, 0.0, 255.0).astype(np.uint8)
        augmented[white] = 255
        return np.ascontiguousarray(augmented.reshape(original_shape))


__all__ = ["GeometricOrientation", "HEStainAugmentation"]
