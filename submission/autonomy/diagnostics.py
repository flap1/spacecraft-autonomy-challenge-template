"""Single-thruster calibration and heater bookkeeping for adaptive planning."""

from __future__ import annotations

from spacecraft_autonomy_challenge import (
    Commander,
    MissionBrief,
    PropulsionMode,
    Telemetry,
)

from .linear import LPStrategy
from .models import (
    PlanAction,
    firing_thruster,
)


class DiagnosticStrategy(LPStrategy):
    """Plan informative pulses and record the exact interval actually executed."""

    def initialize(
        self, brief: MissionBrief, telemetry: Telemetry, commander: Commander
    ) -> None:
        self.reference = brief.nominal_parameters
        self.heat = {
            mode: firing_thruster(self.reference, mode).heat_w
            for mode in (PropulsionMode.THRUSTER_1, PropulsionMode.THRUSTER_2)
        }
        self.probes = list(self.heat)
        self.last_diagnosis = 0.0
        self.diagnosis_count = 0
        self.last_actions: list[PlanAction] = []
        self.used_heater = False
        super().initialize(brief, telemetry, commander)

    def plan(self, telemetry: Telemetry, interval: float) -> list[PlanAction]:
        actions = None
        if self.probes:
            mode = self.probes.pop(0)
            burn = self.maximum_burn(
                telemetry.propulsion_temperature_c,
                telemetry.battery_energy_wh,
                telemetry.wheel_momentum_nms,
                mode,
                interval,
            )
            acceleration = self.acceleration[mode]
            if acceleration > 1e-12:
                burn = min(
                    burn, (self.brief.target_delta_v_mps - self.dv) / acceleration
                )
            if burn >= min(10.0, interval):
                actions = [PlanAction(0.0, mode, burn, burn)]
                if burn < interval:
                    actions.append(
                        PlanAction(burn, PropulsionMode.OFF, 0.0, interval - burn)
                    )
                self.diagnosis_count += 1
        if actions is None:
            actions = super().plan(telemetry, interval)
        self.last_actions = actions
        self.used_heater = (
            not any(action[2] > 1e-6 for action in actions)
            and telemetry.propulsion_temperature_c
            < self.reference.min_firing_temperature_c + 0.2
            and telemetry.battery_energy_wh
            > self.brief.limits.min_battery_energy_wh + 5
        )
        return actions
