"""Adaptive beam-search MPC; only the first planned interval is executed.

Predict to the mission deadline with 5-second steps in the first interval and
coarser steps thereafter. Execute all first-interval actions, then replan.
Acceleration is estimated from wheel momentum, assuming
that thrust and wheel torque share the nominal proportionality. A sudden delta-v
sensor offset is corrected using that independent estimate.
"""

from __future__ import annotations

import math
from typing import ClassVar

from spacecraft_autonomy_challenge import (
    Commander,
    DeclareCompletion,
    MissionBrief,
    PropulsionMode,
    SetHeater,
    SetMomentumDump,
    SetPropulsionMode,
    Strategy,
    Telemetry,
)

from .models import (
    PlanAction,
    PredictedState,
    firing_thruster,
    validate_plan,
)
from .thermal import temperature_after


class BeamStrategy(Strategy):
    CONTROL_STEP_S: ClassVar[float] = 5.0
    BEAM_WIDTH: ClassVar[int] = 16
    THERMAL_MARGIN_C: ClassVar[float] = 0.5
    MOMENTUM_MARGIN_NMS: ClassVar[float] = 0.2
    PACE_FINISH: ClassVar[bool] = False
    REFERENCE_TIME_S: ClassVar[float] = 10660.0

    def initialize(
        self, brief: MissionBrief, telemetry: Telemetry, commander: Commander
    ) -> None:
        self.dump_context = False
        self.thermal_margin_c = self.THERMAL_MARGIN_C
        self.momentum_margin_nms = self.MOMENTUM_MARGIN_NMS
        self.last_dump_momentum = 0.0
        self.last_extra_heat_response = 0.0
        self.last_extra_power_ws = 0.0
        self.brief = brief
        self.previous = telemetry
        self.last_burns = {
            mode: 0.0 for mode in (PropulsionMode.THRUSTER_1, PropulsionMode.THRUSTER_2)
        }
        self.bias = 0.0
        self.dv = telemetry.delta_v_mps
        self.modes = (PropulsionMode.THRUSTER_1, PropulsionMode.THRUSTER_2)
        self.acceleration = {
            mode: firing_thruster(brief.nominal_parameters, mode).acceleration_mps2
            for mode in self.modes
        }
        self.control(telemetry, commander)

    def update(self, telemetry: Telemetry, commander: Commander) -> None:
        self.observe(telemetry)
        observed = telemetry.delta_v_mps - self.previous.delta_v_mps
        momentum = telemetry.wheel_momentum_nms - (
            self.previous.wheel_momentum_nms + self.last_dump_momentum
        )
        p = self.brief.nominal_parameters
        ratios = {
            mode: firing_thruster(p, mode).wheel_momentum_rate_nms_per_s
            / max(firing_thruster(p, mode).acceleration_mps2, 1e-12)
            for mode in self.modes
        }
        predicted = {
            mode: self.acceleration[mode] * self.last_burns[mode] for mode in self.modes
        }
        expected = sum(predicted.values())
        expected_momentum = sum(ratios[m] * predicted[m] for m in self.modes)
        increment = observed
        if abs(observed - expected) > max(0.02, 0.5 * expected):
            # Sensor-jump correction is underdetermined if thrust changes at the
            # same time. Attribute the momentum residual to the dominant burn.
            dominant = max(self.modes, key=lambda m: self.last_burns[m])
            ratio = ratios[dominant]
            correction = (momentum - expected_momentum) / ratio if ratio else 0.0
            increment = max(0.0, expected + correction)
            self.bias += observed - increment
        self.dv += increment

        # Two observations (delta-v sum and signed momentum) identify the two
        # thrusters even when their torques cancel within the same 60 seconds.
        first, second = self.modes
        denominator = ratios[first] - ratios[second]
        if abs(denominator) > 1e-12:
            first_dv = (momentum - ratios[second] * increment) / denominator
            contributions = {first: first_dv, second: increment - first_dv}
            for mode in self.modes:
                burn = self.last_burns[mode]
                if burn > 1e-6 and contributions[mode] >= -1e-8:
                    self.acceleration[mode] = max(0.0, contributions[mode] / burn)
        self.control(telemetry, commander)

    def observe(self, telemetry: Telemetry) -> None:
        """Update optional model estimates before thrust estimation and control."""

    def effective_acceleration(self, mode: PropulsionMode) -> float:
        """Planning rate; subclasses may represent an auxiliary actuator."""
        return self.acceleration[mode]

    def temperature(self, temperature: float, heat: float, duration: float) -> float:
        return temperature_after(
            self.brief.nominal_parameters, temperature, heat, duration
        )

    def maximum_burn(
        self,
        temperature: float,
        energy: float,
        momentum: float,
        mode: PropulsionMode,
        interval: float,
    ) -> float:
        p = self.brief.nominal_parameters
        limits = self.brief.limits
        thruster = firing_thruster(p, mode)
        if temperature < p.min_firing_temperature_c + 0.2:
            return 0.0
        duration = interval
        ceiling = limits.max_propulsion_temperature_c - self.thermal_margin_c
        if temperature >= ceiling:
            return 0.0
        conductance = p.thermal_conductance_w_per_k
        if conductance > 0:
            equilibrium = p.sink_temperature_c + thruster.heat_w / conductance
            if equilibrium > ceiling:
                duration = min(
                    duration,
                    p.heat_capacity_j_per_k
                    / conductance
                    * math.log((equilibrium - temperature) / (equilibrium - ceiling)),
                )
        elif thruster.heat_w > 0:
            duration = min(
                duration,
                (ceiling - temperature) * p.heat_capacity_j_per_k / thruster.heat_w,
            )
        drain = p.bus_load_w + thruster.power_w - p.power_generation_w
        if drain > 0:
            duration = min(
                duration, (energy - limits.min_battery_energy_wh - 1) * 3600 / drain
            )
        nominal = thruster.acceleration_mps2
        efficiency = self.acceleration[mode] / nominal if nominal > 0 else 1.0
        torque = thruster.wheel_momentum_rate_nms_per_s * efficiency
        bound = limits.max_abs_wheel_momentum_nms - self.momentum_margin_nms
        if abs(torque) > 1e-12:
            edge = math.copysign(bound, torque)
            duration = min(duration, (edge - momentum) / torque)
        return max(0.0, duration)

    def predict(
        self, state: PredictedState, mode: PropulsionMode, burn: float, interval: float
    ) -> PredictedState:
        dv, temperature, energy, momentum = state
        p = self.brief.nominal_parameters
        generation = p.power_generation_w - p.bus_load_w
        if burn > 0:
            thruster = firing_thruster(p, mode)
            acceleration = self.acceleration[mode]
            dv += acceleration * burn
            temperature = self.temperature(temperature, thruster.heat_w, burn)
            energy += (generation - thruster.power_w) * burn / 3600
            efficiency = acceleration / max(thruster.acceleration_mps2, 1e-12)
            momentum += thruster.wheel_momentum_rate_nms_per_s * efficiency * burn
        coast = interval - burn
        temperature = self.temperature(temperature, 0.0, coast)
        energy = min(p.battery_capacity_wh, energy + generation * coast / 3600)
        return PredictedState(dv, temperature, energy, momentum)

    def plan(self, telemetry: Telemetry, interval: float) -> list[PlanAction]:
        initial = PredictedState(
            self.dv,
            telemetry.propulsion_temperature_c,
            telemetry.battery_energy_wh,
            telemetry.wheel_momentum_nms,
        )
        # Nodes: objective, predicted state, executable prefix, elapsed time.
        beam: list[tuple[float, PredictedState, tuple[PlanAction, ...], float]] = [
            (0.0, initial, (), 0.0)
        ]
        target = self.brief.target_delta_v_mps
        time_left = self.brief.time_limit_s - telemetry.time_s
        fine_steps = math.ceil(interval / self.CONTROL_STEP_S)
        coarse_steps = math.ceil(max(0.0, time_left - interval) / interval)
        for depth in range(fine_steps + coarse_steps):
            candidates = []
            for score, state, prefix, elapsed in beam:
                if state[0] >= target - 1e-9 or elapsed >= time_left:
                    candidates.append((score, state, prefix, elapsed))
                    continue
                if depth < fine_steps:
                    step = min(self.CONTROL_STEP_S, interval - elapsed)
                else:
                    step = self.brief.decision_period_s
                step = min(step, time_left - elapsed)
                actions = [(PropulsionMode.OFF, 0.0)]
                for mode in self.modes:
                    burn = self.maximum_burn(*state[1:], mode, step)
                    acceleration = self.effective_acceleration(mode)
                    if acceleration > 1e-12:
                        burn = min(burn, (target - state[0]) / acceleration)
                    if (
                        self.PACE_FINISH
                        and depth < fine_steps
                        and acceleration > 1e-12
                        and target - self.dv <= max(self.acceleration.values()) * 180
                        and telemetry.time_s + elapsed < self.REFERENCE_TIME_S
                    ):
                        until_reference = (
                            self.REFERENCE_TIME_S - telemetry.time_s - elapsed
                        )
                        burn = min(
                            burn,
                            (target - state[0])
                            * step
                            / max(step, until_reference)
                            / acceleration,
                        )
                    if burn > 1e-6:
                        actions.extend(((mode, burn), (mode, burn * 0.5)))
                for mode, burn in actions:
                    predicted = self.predict(state, mode, burn, step)
                    reached = predicted[0] >= target - 1e-9
                    duration = burn if reached else step
                    # Progress dominates; retain thermal/electrical headroom and
                    # prefer balanced momentum when the thrust gains are equal.
                    value = (
                        predicted[0]
                        - self.dv
                        - 0.0001 * predicted[1]
                        + 0.00001 * predicted[2]
                        - 0.00001 * predicted[3] ** 2
                    )
                    if reached:
                        value += 0.001 * (time_left - elapsed - duration)
                    executable = prefix
                    if depth < fine_steps:
                        executable += (PlanAction(elapsed, mode, burn, step),)
                    candidates.append(
                        (value, predicted, executable, elapsed + duration)
                    )
            candidates.sort(key=lambda node: node[0], reverse=True)
            beam = candidates[: self.BEAM_WIDTH]
            if all(node[1][0] >= target - 1e-9 for node in beam):
                break
        return list(beam[0][2])

    def control(self, telemetry: Telemetry, commander: Commander) -> None:
        now = commander.now_s
        commander.cancel_from(now)
        commander.execute(SetMomentumDump(on=False))
        commander.execute(SetHeater(on=False))
        if self.dv >= self.brief.target_delta_v_mps - 1e-8:
            commander.execute(SetPropulsionMode(PropulsionMode.OFF))
            commander.execute(DeclareCompletion())
            return
        interval = min(self.brief.decision_period_s, self.brief.time_limit_s - now)
        actions = self.plan(telemetry, interval)
        validate_plan(actions, interval)
        self.last_burns = {mode: 0.0 for mode in self.modes}
        remaining = self.brief.target_delta_v_mps - self.dv
        scheduled_mode = None
        total_gain = 0.0
        for offset, mode, burn, step in actions:
            if not isinstance(mode, PropulsionMode):
                raise TypeError("beam execution received a non-propulsive action")
            if burn <= 1e-6:
                mode, burn = PropulsionMode.OFF, 0.0
            if mode != scheduled_mode:
                commander.schedule(now + offset, SetPropulsionMode(mode))
                scheduled_mode = mode
            if burn <= 0:
                continue
            self.last_burns[mode] += burn
            total_gain += self.acceleration[mode] * burn
            stop = now + offset + burn
            if total_gain >= remaining - 1e-9:
                commander.schedule(stop, SetPropulsionMode(PropulsionMode.OFF))
                commander.schedule(stop, DeclareCompletion())
                break
            if burn < step - 1e-8:
                commander.schedule(stop, SetPropulsionMode(PropulsionMode.OFF))
                scheduled_mode = PropulsionMode.OFF
        if not any(self.last_burns.values()):
            p = self.brief.nominal_parameters
            if (
                telemetry.propulsion_temperature_c < p.min_firing_temperature_c + 0.2
                and telemetry.battery_energy_wh
                > self.brief.limits.min_battery_energy_wh + 5
            ):
                commander.execute(SetHeater(on=True))
        self.previous = telemetry
