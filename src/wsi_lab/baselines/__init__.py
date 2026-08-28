"""Non-learning policies used as WSI Lab paper baselines."""

from wsi_lab.baselines.base import BaselinePolicy
from wsi_lab.baselines.confidence import ConfidenceThresholdPolicy
from wsi_lab.baselines.fixed_budget import FixedBudgetPolicy
from wsi_lab.baselines.random import RandomPolicy
from wsi_lab.baselines.raster import RasterPolicy
from wsi_lab.baselines.sweep import SweepProtocol, evaluate_navigation_sweep

__all__ = [
    "BaselinePolicy",
    "ConfidenceThresholdPolicy",
    "FixedBudgetPolicy",
    "RandomPolicy",
    "RasterPolicy",
    "SweepProtocol",
    "evaluate_navigation_sweep",
]
