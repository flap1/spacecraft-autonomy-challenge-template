"""Research-inspired adaptive economic MPC with event-triggered identification.

LP handles score-aware battery pacing when thrusters are symmetric. A numerical
MPC handles asymmetric thrust. Thermal regression uses the actual timed pulses,
including their exponential cooling weights. Single-thruster probes are requested
only when observations reveal a model change or poor identifiability.

Safety reserves cover a bounded inter-observation heat/torque change; they are
engineering bounds, not a general tube-MPC proof for arbitrary hidden scenarios.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import ClassVar

import numpy as np
from spacecraft_autonomy_challenge import (
    Commander,
    MissionBrief,
    Telemetry,
)

from .diagnostics import DiagnosticStrategy
from .models import (
    FloatArray,
    firing_thruster,
)
from .thermal import observation


class EconomicStrategy(DiagnosticStrategy):
    """Fit thermal/power models and use them for score-aware planning."""

    USE_BEAM_ON_ASYMMETRY: ClassVar[bool] = True
    LP_TERMINAL_WINDOW_S: ClassVar[float] = 600.0
    LEARN_THERMAL: ClassVar[bool] = True
    ADAPT_RESERVES: ClassVar[bool] = True
    THERMAL_MARGIN_C: ClassVar[float] = 0.5
    MOMENTUM_MARGIN_NMS: ClassVar[float] = 0.2
    THERMAL_CALIBRATED_MARGIN_C: ClassVar[float] = 0.30
    MOMENTUM_CALIBRATED_MARGIN_NMS: ClassVar[float] = 0.08

    def initialize(
        self, brief: MissionBrief, telemetry: Telemetry, commander: Commander
    ) -> None:
        self.thermal_data: deque[tuple[FloatArray, float]] = deque(maxlen=12)
        self.observations = 0
        self.net_generation_w = (
            brief.nominal_parameters.power_generation_w
            - brief.nominal_parameters.bus_load_w
        )
        self.reset_count = 0
        self.last_thermal_residual = 0.0
        self.last_model_change = -math.inf
        super().initialize(brief, telemetry, commander)

    def prefer_beam(self) -> bool:
        """Heat asymmetry benefits from global resource allocation."""
        heats = tuple(self.heat.values())
        if max(heats) - min(heats) > 0.15 * max(max(heats), 1e-12):
            return False
        return super().prefer_beam()

    def observe(self, telemetry: Telemetry) -> None:
        """Learn from the last executed interval before planning the next one."""
        self._observe_thermal(telemetry)
        self._observe_power(telemetry)

    def _observe_thermal(self, telemetry: Telemetry) -> None:
        p = self.reference
        capacity = p.heat_capacity_j_per_k
        weights, observed_heat = observation(
            p,
            self.previous,
            telemetry,
            self.last_actions,
            self.modes,
            heater_on=self.used_heater,
            extra_heat_response=self.last_extra_heat_response,
        )
        phi = np.array(weights)
        y = observed_heat
        estimate = np.array([self.heat[mode] for mode in self.modes])
        residual = y - phi @ estimate
        self.last_thermal_residual = float(residual)
        if phi @ phi > 1e-10 and self.LEARN_THERMAL:
            if abs(residual) > 0.035:
                # Old samples describe the pre-fault plant and must not dominate.
                self.thermal_data.clear()
                self.reset_count += 1
                self.last_model_change = telemetry.time_s
                self.probes = list(self.modes)
            self.thermal_data.append((phi, y))
            matrix = np.stack([sample[0] for sample in self.thermal_data])
            values = np.array([sample[1] for sample in self.thermal_data])
            # A weak prior retains the unobserved direction, avoiding arbitrary
            # redistribution of heat between indistinguishable mixed burns.
            ridge = 1e-8
            gram = matrix.T @ matrix
            estimate = np.linalg.solve(
                gram + ridge * np.eye(2), matrix.T @ values + ridge * estimate
            )
            estimate = np.clip(
                estimate, 0, 3 * max(p.thruster_1.heat_w, p.thruster_2.heat_w)
            )
            for mode, heat in zip(self.modes, estimate, strict=True):
                self.heat[mode] = float(heat)
            if (
                np.linalg.eigvalsh(gram)[0] < 1e-7
                and abs(residual) > 0.01
                and telemetry.time_s - self.last_diagnosis >= 180
            ):
                self.probes = list(self.modes)
                self.last_diagnosis = telemetry.time_s
            self.brief = replace(
                self.brief,
                nominal_parameters=replace(
                    p,
                    thruster_1=replace(p.thruster_1, heat_w=self.heat[self.modes[0]]),
                    thruster_2=replace(p.thruster_2, heat_w=self.heat[self.modes[1]]),
                ),
            )
        self.observations += 1
        if self.ADAPT_RESERVES and self.observations >= 2:
            # Reserve for an unobserved 20% heat change over a 60 s interval,
            # plus integration/estimation allowance. Re-expand after surprises.
            interval = self.brief.decision_period_s
            heat_jump = 0.20 * max(self.heat.values()) * interval / capacity
            residual_allowance = min(0.5, 2 * abs(residual))
            self.thermal_margin_c = max(
                self.THERMAL_CALIBRATED_MARGIN_C,
                heat_jump + 0.05 + residual_allowance,
            )
            self.momentum_margin_nms = self.MOMENTUM_CALIBRATED_MARGIN_NMS

    def _observe_power(self, telemetry: Telemetry) -> None:
        p = self.reference
        dt = telemetry.time_s - self.previous.time_s
        if (
            dt > 0
            and telemetry.battery_energy_wh < p.battery_capacity_wh - 0.01
            and self.previous.battery_energy_wh < p.battery_capacity_wh - 0.01
        ):
            load = (
                sum(
                    firing_thruster(p, mode).power_w * self.last_burns[mode]
                    for mode in self.modes
                )
                / dt
            )
            load += self.last_extra_power_ws / dt
            if self.used_heater:
                load += p.heater_power_w
            observed_generation = (
                telemetry.battery_energy_wh - self.previous.battery_energy_wh
            ) * 3600 / dt + load
            self.net_generation_w = observed_generation
        model = self.brief.nominal_parameters
        self.brief = replace(
            self.brief,
            nominal_parameters=replace(
                model, power_generation_w=model.bus_load_w + self.net_generation_w
            ),
        )
