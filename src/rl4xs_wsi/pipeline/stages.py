"""Pipeline stage names and lifecycle decisions."""

from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    DATASET = "dataset"
    BELIEF_DATA = "belief-data"
    SPLIT = "split"
    TISSUE = "tissue"
    PREPROCESSING = "preprocessing"
    BELIEF = "belief"
    PPO = "ppo"


class StageAction(StrEnum):
    SKIP = "SKIP"
    BUILD = "BUILD"
    REBUILD = "REBUILD"
    RESUME = "RESUME"
    TRAIN = "TRAIN"


STAGE_ORDER = (
    Stage.DATASET,
    Stage.SPLIT,
    Stage.TISSUE,
    Stage.PREPROCESSING,
    Stage.BELIEF_DATA,
    Stage.BELIEF,
    Stage.PPO,
)
