"""Conservative policy and recovery behavior for practice evaluation.

The beam planner is the fallback for the adaptive planner. This policy uses
adaptive planning and momentum dumping with a four-degree thermal reserve.
"""

from __future__ import annotations

from typing import ClassVar

from spacecraft_autonomy_challenge import (
    PropulsionMode,
    Telemetry,
)

from .dump import DumpEconomicStrategy
from .models import (
    PlanAction,
    firing_thruster,
)


class ConservativeStrategy(DumpEconomicStrategy):
    THERMAL_MARGIN_C: ClassVar[float] = 4.0
    THERMAL_CALIBRATED_MARGIN_C: ClassVar[float] = 4.0
    MOMENTUM_MARGIN_NMS: ClassVar[float] = 0.5
    MOMENTUM_CALIBRATED_MARGIN_NMS: ClassVar[float] = 0.5
    WARMUP_MARGIN_C: ClassVar[float] = 3.0
    HEATER_BATTERY_RESERVE_WH: ClassVar[float] = 5.0
    STALLED_THRUST_FRACTION: ClassVar[float] = 0.02
    RETRY_DURATION_S: ClassVar[float] = 10.0

    def plan(self, telemetry: Telemetry, interval: float) -> list[PlanAction]:
        p = self.brief.nominal_parameters
        cold = (
            telemetry.propulsion_temperature_c
            < p.min_firing_temperature_c + self.WARMUP_MARGIN_C
        )
        if cold:
            self.last_actions = [PlanAction(0.0, PropulsionMode.OFF, 0.0, interval)]
            self.used_heater = (
                telemetry.battery_energy_wh
                > self.brief.limits.min_battery_energy_wh
                + self.HEATER_BATTERY_RESERVE_WH
            )
            return self.last_actions

        stalled = [
            mode
            for mode in self.modes
            if self.acceleration[mode]
            < self.STALLED_THRUST_FRACTION
            * firing_thruster(self.reference, mode).acceleration_mps2
        ]
        if stalled and max(self.acceleration.values()) < 1e-7:
            # A cold-gated thruster must be tried again after warming. Do not
            # declare completion on this uncalibrated, short diagnostic pulse.
            mode = stalled[0]
            nominal = firing_thruster(self.reference, mode).acceleration_mps2
            remaining = self.brief.target_delta_v_mps - self.dv
            burn = min(
                self.RETRY_DURATION_S, interval, remaining / max(2 * nominal, 1e-12)
            )
            self.acceleration[mode] = nominal * 0.25
            self.last_actions = [PlanAction(0.0, mode, burn, burn)]
            if burn < interval:
                self.last_actions.append(
                    PlanAction(burn, PropulsionMode.OFF, 0.0, interval - burn)
                )
            self.used_heater = False
            return self.last_actions
        return super().plan(telemetry, interval)
