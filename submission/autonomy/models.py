"""Typed values shared by prediction, planning and command execution."""

import math
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

from spacecraft_autonomy_challenge import (
    PropulsionMode,
    SpacecraftParameters,
    ThrusterParameters,
)


class AuxiliaryMode(Enum):
    """Non-propulsive actions supported by the execution layer."""

    MOMENTUM_DUMP = "momentum_dump"


DUMP = AuxiliaryMode.MOMENTUM_DUMP


class PlanAction(NamedTuple):
    """Fire/dump at an offset, then coast for the rest of the interval.

    Offsets and durations are relative seconds, not absolute simulation times.
    A zero burn with mode OFF represents a coast interval.
    """

    offset_s: float
    mode: PropulsionMode | AuxiliaryMode
    burn_s: float
    interval_s: float


class PredictedState(NamedTuple):
    delta_v_mps: float
    temperature_c: float
    battery_wh: float
    momentum_nms: float


class ActuatorRates(NamedTuple):
    acceleration_mps2: float
    torque_nms_per_s: float
    heat_w: float
    power_w: float


def firing_thruster(
    parameters: SpacecraftParameters, mode: PropulsionMode
) -> ThrusterParameters:
    """Resolve a firing mode; OFF must never be treated as a thruster."""
    thruster = parameters.thruster(mode)
    if thruster is None:
        raise ValueError("a firing thruster is required, got OFF")
    return thruster


def validate_plan(actions: list[PlanAction], interval_s: float) -> None:
    """Reject corrupt solver output before it becomes spacecraft commands."""
    if not math.isfinite(interval_s) or interval_s < 0:
        raise ValueError("observation interval must be finite and nonnegative")
    previous_end = 0.0
    tolerance = 1e-6
    for action in actions:
        values = (action.offset_s, action.burn_s, action.interval_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("plan contains a non-finite time")
        if min(values) < 0 or action.burn_s > action.interval_s + tolerance:
            raise ValueError("plan contains a negative or inconsistent duration")
        if action.offset_s < previous_end - tolerance:
            raise ValueError("plan intervals overlap or are out of order")
        previous_end = action.offset_s + action.interval_s
        if previous_end > interval_s + tolerance:
            raise ValueError("plan extends beyond the next observation")
