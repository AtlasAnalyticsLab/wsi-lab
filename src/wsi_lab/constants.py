"""Protocol constants for WSILab-v0."""

from __future__ import annotations

from typing import Final

PACKAGE_VERSION: Final = "0.1.0"
SCHEMA_VERSION: Final = 1
ENV_ID: Final = "WSILab-v0"

NAVIGATION_MAGNIFICATIONS: Final[tuple[float, ...]] = (0.625, 1.25, 2.5, 5.0, 10.0, 20.0)
BELIEF_MAGNIFICATIONS: Final[tuple[float, ...]] = (2.5, 5.0, 10.0, 20.0)
LEVEL0_FOOTPRINTS: Final[dict[float, int]] = {
    0.625: 16384,
    1.25: 8192,
    2.5: 4096,
    5.0: 2048,
    10.0: 1024,
    20.0: 512,
}
# Each deterministic adapter slide index owns one non-overlapping int64 ID
# block.  The generous capacity permits arbitrary sparse preprocessing without
# inspecting preceding slides to discover their dimensions.
CELL_ID_SLIDE_BLOCK_SIZE: Final = 1 << 40
NUM_ACTIONS: Final = 11
DEFAULT_FEATURE_KEY: Final = "features/uni_v2"
DEFAULT_COORDINATE_KEY: Final = "coords"
DEFAULT_PASSPORT_KEY: Final = "passports"


def magnification_token(value: float) -> str:
    """Return the stable path token used by native embedding directories."""
    return str(int(value)) if float(value).is_integer() else str(value)
