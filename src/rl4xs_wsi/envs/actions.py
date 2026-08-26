"""Stable action enumeration and geometry relation helpers."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    PAN_UP = 0
    PAN_DOWN = 1
    PAN_LEFT = 2
    PAN_RIGHT = 3
    ZOOM_IN_TL = 4
    ZOOM_IN_TR = 5
    ZOOM_IN_BL = 6
    ZOOM_IN_BR = 7
    ZOOM_OUT = 8
    DIAGNOSE_CANCEROUS = 9
    DIAGNOSE_BENIGN = 10


PAN_ACTIONS = (Action.PAN_UP, Action.PAN_DOWN, Action.PAN_LEFT, Action.PAN_RIGHT)
ZOOM_IN_ACTIONS = (
    Action.ZOOM_IN_TL,
    Action.ZOOM_IN_TR,
    Action.ZOOM_IN_BL,
    Action.ZOOM_IN_BR,
)
NAVIGATION_ACTIONS = PAN_ACTIONS + ZOOM_IN_ACTIONS + (Action.ZOOM_OUT,)
DIAGNOSIS_ACTIONS = (Action.DIAGNOSE_CANCEROUS, Action.DIAGNOSE_BENIGN)
