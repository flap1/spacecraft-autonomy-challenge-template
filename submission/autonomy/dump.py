"""Hybrid economic MPC with a globally planned healthy-thruster/dump mode."""

from __future__ import annotations

import math

from spacecraft_autonomy_challenge import (
    Commander,
    DeclareCompletion,
    MissionBrief,
    PropulsionMode,
    SetHeater,
    SetMomentumDump,
    SetPropulsionMode,
    Telemetry,
)

from .economic import EconomicStrategy
from .models import (
    DUMP,
    ActuatorRates,
    PlanAction,
    PredictedState,
    validate_plan,
)
from .thermal import heat_response


class DumpEconomicStrategy(EconomicStrategy):
    def initialize(
        self, brief: MissionBrief, telemetry: Telemetry, commander: Commander
    ) -> None:
        self.dump_context = False
        self.dump_intervals = 0
        self.last_dump_momentum = 0.0
        self.last_extra_heat_response = 0.0
        self.last_extra_power_ws = 0.0
        super().initialize(brief, telemetry, commander)

    def actuator_terms(self, mode: PropulsionMode) -> ActuatorRates:
        if self.dump_context and mode == self.dump_slot:
            p = self.brief.nominal_parameters
            return ActuatorRates(
                0.0,
                -self.dump_sign * p.momentum_dump_rate_nms_per_s,
                p.momentum_dump_heat_w,
                p.momentum_dump_power_w,
            )
        return super().actuator_terms(mode)

    def momentum_bounds(self, bound: float) -> tuple[float, float]:
        if self.dump_context:
            return (0, bound) if self.dump_sign > 0 else (-bound, 0)
        return super().momentum_bounds(bound)

    def maximum_burn(
        self,
        temperature: float,
        energy: float,
        momentum: float,
        mode: PropulsionMode,
        interval: float,
    ) -> float:
        if self.dump_context and mode == self.dump_slot:
            p = self.brief.nominal_parameters
            if momentum * self.dump_sign <= 0 or p.momentum_dump_rate_nms_per_s <= 0:
                return 0.0
            duration = min(interval, abs(momentum) / p.momentum_dump_rate_nms_per_s)
            drain = p.bus_load_w + p.momentum_dump_power_w - p.power_generation_w
            if drain > 0:
                duration = min(
                    duration,
                    max(0.0, energy - (self.brief.limits.min_battery_energy_wh + 1))
                    * 3600
                    / drain,
                )
            # The published dump heats less than the passive cooling at the
            # thermal boundary; reject unfamiliar hot-dump behavior.
            if self.temperature(temperature, p.momentum_dump_heat_w, duration) > (
                self.brief.limits.max_propulsion_temperature_c - self.thermal_margin_c
            ):
                return 0.0
            return duration
        return super().maximum_burn(temperature, energy, momentum, mode, interval)

    def predict(
        self, state: PredictedState, mode: PropulsionMode, burn: float, interval: float
    ) -> PredictedState:
        if self.dump_context and mode == self.dump_slot:
            dv, temperature, energy, momentum = state
            p = self.brief.nominal_parameters
            temperature = self.temperature(temperature, p.momentum_dump_heat_w, burn)
            temperature = self.temperature(temperature, 0.0, interval - burn)
            energy += (
                (p.power_generation_w - p.bus_load_w) * interval
                - p.momentum_dump_power_w * burn
            ) / 3600
            momentum -= self.dump_sign * min(
                abs(momentum), p.momentum_dump_rate_nms_per_s * burn
            )
            return PredictedState(
                dv, temperature, min(energy, p.battery_capacity_wh), momentum
            )
        return super().predict(state, mode, burn, interval)

    def dump_is_beneficial(self, telemetry: Telemetry) -> bool:
        p = self.brief.nominal_parameters
        healthy = max(self.modes, key=lambda mode: self.acceleration[mode])
        weak = next(mode for mode in self.modes if mode != healthy)
        acceleration, torque, heat, _ = self.actuator_terms(healthy)
        weak_acc, weak_torque, weak_heat, _ = self.actuator_terms(weak)
        rate = p.momentum_dump_rate_nms_per_s
        if rate <= 0 or torque * weak_torque > 0 or acceleration <= 0:
            return False
        sign = math.copysign(1.0, torque)
        if telemetry.wheel_momentum_nms * sign < 0:
            return False
        healthy_fraction = rate / (rate + abs(torque))
        counter_fraction = abs(weak_torque) / (abs(torque) + abs(weak_torque))
        cooling = p.thermal_conductance_w_per_k * (
            self.brief.limits.max_propulsion_temperature_c
            - self.thermal_margin_c
            - p.sink_temperature_c
        )
        dump_heat = (
            healthy_fraction * heat + (1 - healthy_fraction) * p.momentum_dump_heat_w
        )
        two_heat = counter_fraction * heat + (1 - counter_fraction) * weak_heat
        dump_speed = (
            healthy_fraction * acceleration * min(1.0, cooling / max(dump_heat, 1e-9))
        )
        two_speed = counter_fraction * acceleration + (1 - counter_fraction) * weak_acc
        two_speed *= min(1.0, cooling / max(two_heat, 1e-9))
        if dump_speed <= 1.03 * two_speed:
            return False
        self.dump_slot, self.healthy, self.dump_sign = weak, healthy, sign
        return True

    def prefer_beam(self) -> bool:
        return False if self.dump_context else super().prefer_beam()

    def plan(self, telemetry: Telemetry, interval: float) -> list[PlanAction]:
        if not self.dump_is_beneficial(telemetry):
            return super().plan(telemetry, interval)
        self.dump_context = True
        try:
            actions = self._plan_linear(telemetry, interval)
        finally:
            self.dump_context = False
        actions = [
            PlanAction(
                offset,
                DUMP if mode == self.dump_slot and burn > 0 else mode,
                burn,
                step,
            )
            for offset, mode, burn, step in actions
        ]
        self.last_actions = actions
        self.used_heater = False
        return actions

    def control(self, telemetry: Telemetry, commander: Commander) -> None:
        now = commander.now_s
        commander.cancel_from(now)
        commander.execute(SetMomentumDump(on=False))
        commander.execute(SetHeater(on=False))
        self.last_dump_momentum = 0.0
        self.last_extra_heat_response = 0.0
        self.last_extra_power_ws = 0.0
        if self.dv >= self.brief.target_delta_v_mps - 1e-8:
            commander.execute(SetPropulsionMode(PropulsionMode.OFF))
            commander.execute(DeclareCompletion())
            return
        interval = min(self.brief.decision_period_s, self.brief.time_limit_s - now)
        actions = self.plan(telemetry, interval)
        validate_plan(actions, interval)
        self.last_burns = dict.fromkeys(self.modes, 0.0)
        remaining = self.brief.target_delta_v_mps - self.dv
        total_gain = 0.0
        momentum = telemetry.wheel_momentum_nms
        p = self.brief.nominal_parameters
        for offset, mode, burn, step in actions:
            at = now + offset
            commander.schedule(at, SetMomentumDump(on=False))
            if mode == DUMP and burn > 1e-7:
                commander.schedule(at, SetPropulsionMode(PropulsionMode.OFF))
                commander.schedule(at, SetMomentumDump(on=True))
                commander.schedule(at + burn, SetMomentumDump(on=False))
                change = -math.copysign(
                    min(abs(momentum), p.momentum_dump_rate_nms_per_s * burn), momentum
                )
                momentum += change
                self.last_dump_momentum += change
                self.last_extra_power_ws += p.momentum_dump_power_w * burn
                self.last_extra_heat_response += p.momentum_dump_heat_w * heat_response(
                    p, offset, burn, interval
                )
                self.dump_intervals += 1
                continue
            if not isinstance(mode, PropulsionMode):
                raise TypeError("unsupported auxiliary action")
            if burn <= 1e-7:
                mode, burn = PropulsionMode.OFF, 0.0
            commander.schedule(at, SetPropulsionMode(mode))
            if burn > 0:
                self.last_burns[mode] += burn
                total_gain += self.acceleration[mode] * burn
                momentum += self.actuator_terms(mode)[1] * burn
                stop = at + burn
                if total_gain >= remaining - 1e-9:
                    commander.schedule(stop, SetPropulsionMode(PropulsionMode.OFF))
                    commander.schedule(stop, DeclareCompletion())
                    break
                if burn < step - 1e-8:
                    commander.schedule(stop, SetPropulsionMode(PropulsionMode.OFF))
        if self.used_heater:
            commander.execute(SetHeater(on=True))
        self.previous = telemetry
