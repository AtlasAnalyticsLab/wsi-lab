"""Non-learning policies used as RL4XS WSI paper baselines."""

from rl4xs_wsi.baselines.base import BaselinePolicy
from rl4xs_wsi.baselines.confidence import ConfidenceThresholdPolicy
from rl4xs_wsi.baselines.fixed_budget import FixedBudgetPolicy
from rl4xs_wsi.baselines.random import RandomPolicy
from rl4xs_wsi.baselines.raster import RasterPolicy

__all__ = [
    "BaselinePolicy",
    "ConfidenceThresholdPolicy",
    "FixedBudgetPolicy",
    "RandomPolicy",
    "RasterPolicy",
]
