"""Exact continuous thermal predictions and timed-input observation weights."""

import math

from spacecraft_autonomy_challenge import (
    PropulsionMode,
    SpacecraftParameters,
    Telemetry,
)

from .models import PlanAction


def temperature_after(
    parameters: SpacecraftParameters,
    temperature_c: float,
    heat_w: float,
    duration_s: float,
) -> float:
    conductance = parameters.thermal_conductance_w_per_k
    if conductance <= 0:
        return temperature_c + heat_w * duration_s / parameters.heat_capacity_j_per_k
    equilibrium = parameters.sink_temperature_c + heat_w / conductance
    decay = math.exp(-conductance * duration_s / parameters.heat_capacity_j_per_k)
    return equilibrium + (temperature_c - equilibrium) * decay


def heat_response(
    parameters: SpacecraftParameters, offset_s: float, burn_s: float, interval_s: float
) -> float:
    """Temperature response per watt of a timed constant-power pulse."""
    end = min(interval_s, offset_s + burn_s)
    duration = max(0.0, end - offset_s)
    k = parameters.thermal_conductance_w_per_k
    capacity = parameters.heat_capacity_j_per_k
    if k == 0:
        return duration / capacity
    return (
        math.exp(-k * (interval_s - end) / capacity)
        * -math.expm1(-k * duration / capacity)
        / k
    )


def observation(
    parameters: SpacecraftParameters,
    previous: Telemetry,
    telemetry: Telemetry,
    actions: list[PlanAction],
    modes: tuple[PropulsionMode, ...],
    *,
    heater_on: bool,
    extra_heat_response: float = 0.0,
) -> tuple[list[float], float]:
    """Return pulse coefficients and observed heat contribution for regression."""
    dt = telemetry.time_s - previous.time_s
    passive = temperature_after(parameters, previous.propulsion_temperature_c, 0, dt)
    passive += extra_heat_response
    if heater_on:
        passive += parameters.heater_power_w * heat_response(parameters, 0, dt, dt)
    weights = dict.fromkeys(modes, 0.0)
    for action in actions:
        if action.burn_s > 0 and action.mode in weights:
            weights[action.mode] += heat_response(
                parameters, action.offset_s, action.burn_s, dt
            )
    return [
        weights[mode] for mode in modes
    ], telemetry.propulsion_temperature_c - passive
