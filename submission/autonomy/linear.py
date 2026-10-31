"""Global LP with a constrained 5-second execution layer.

Burn times are continuous LP variables; battery, temperature and momentum are
propagated across the complete horizon. Only the next observation interval is
executed, using local safety limits to convert duty cycles into actual pulses.
"""

from __future__ import annotations

import math
import time
import warnings
from enum import IntEnum
from typing import ClassVar

import numpy as np
from scipy.optimize import OptimizeWarning, linprog
from scipy.sparse import lil_matrix
from spacecraft_autonomy_challenge import (
    Commander,
    MissionBrief,
    PropulsionMode,
    Telemetry,
)

from .beam import BeamStrategy
from .models import (
    ActuatorRates,
    FloatArray,
    PlanAction,
    PredictedState,
    firing_thruster,
)


class Column(IntEnum):
    """Variable layout of one LP interval."""

    FIRST_BURN = 0
    SECOND_BURN = 1
    TEMPERATURE = 2
    BATTERY = 3
    MOMENTUM = 4
    SPILLED_ENERGY = 5


VARIABLES_PER_STEP = len(Column)


class LPStrategy(BeamStrategy):
    """Allocate global resources, then realize the next interval as pulses."""

    USE_BEAM_ON_ASYMMETRY: ClassVar[bool] = False
    LP_TERMINAL_WINDOW_S: ClassVar[float] = math.inf

    def initialize(
        self, brief: MissionBrief, telemetry: Telemetry, commander: Commander
    ) -> None:
        self.cpu_started = time.process_time()
        self.lp_cpu_limit = 0.6 * (
            5 + 0.1 * brief.time_limit_s / brief.decision_period_s
        )
        self.planned_end = min(self.REFERENCE_TIME_S, brief.time_limit_s)
        self.lp_calls = 0
        self.lp_failures = 0
        super().initialize(brief, telemetry, commander)

    def actuator_terms(self, mode: PropulsionMode) -> ActuatorRates:
        thruster = firing_thruster(self.brief.nominal_parameters, mode)
        acceleration = self.acceleration[mode]
        torque = (
            thruster.wheel_momentum_rate_nms_per_s
            * acceleration
            / max(thruster.acceleration_mps2, 1e-12)
        )
        return ActuatorRates(acceleration, torque, thruster.heat_w, thruster.power_w)

    def momentum_bounds(self, bound: float) -> tuple[float, float]:
        return -bound, bound

    def solve(
        self, telemetry: Telemetry, deadline: float
    ) -> tuple[FloatArray, list[float]] | None:
        p = self.brief.nominal_parameters
        limits = self.brief.limits
        duration = deadline - telemetry.time_s
        if duration <= 0:
            return None
        fine_duration = min(duration, self.brief.decision_period_s)
        steps = []
        elapsed = 0.0
        while elapsed < duration - 1e-7:
            step = self.CONTROL_STEP_S if elapsed < fine_duration - 1e-7 else 60.0
            step = (
                min(step, duration - elapsed, fine_duration - elapsed)
                if (elapsed < fine_duration - 1e-7)
                else min(step, duration - elapsed)
            )
            steps.append(step)
            elapsed += step
        n = len(steps)
        # Per interval: burn1, burn2, endpoint T/E/H, spilled solar energy.
        size = VARIABLES_PER_STEP * n + 1
        minimum_energy = size - 1
        equalities = lil_matrix((3 * n + 1, size))
        rhs = np.zeros(3 * n + 1)
        inequalities = lil_matrix((2 * n, size))
        upper = np.zeros(2 * n)
        objective = np.zeros(size)
        objective[minimum_energy] = -1.0
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * size
        elapsed = 0.0
        initial = (
            telemetry.propulsion_temperature_c,
            telemetry.battery_energy_wh,
            telemetry.wheel_momentum_nms,
        )
        for i, step in enumerate(steps):
            base = VARIABLES_PER_STEP * i
            alpha = math.exp(
                -p.thermal_conductance_w_per_k * step / p.heat_capacity_j_per_k
            )
            thermal_gain = (
                (1 - alpha) / p.thermal_conductance_w_per_k / step
                if p.thermal_conductance_w_per_k
                else 1 / p.heat_capacity_j_per_k
            )
            for j, mode in enumerate(self.modes):
                acceleration, torque, heat, power = self.actuator_terms(mode)
                equalities[3 * i, base + j] = -thermal_gain * heat
                equalities[3 * i + 1, base + j] = power / 3600
                equalities[3 * i + 2, base + j] = -torque
                equalities[3 * n, base + j] = acceleration
                inequalities[2 * i, base + j] = 1
                bounds[base + j] = (0, step)
                # Prefer early progress only after optimizing battery minimum.
                objective[base + j] = 1e-5 * elapsed / duration
            for j, factor in enumerate((alpha, 1.0, 1.0)):
                equalities[3 * i + j, base + Column.TEMPERATURE + j] = 1
                if i:
                    equalities[
                        3 * i + j, base - VARIABLES_PER_STEP + Column.TEMPERATURE + j
                    ] = -factor
                else:
                    rhs[3 * i + j] = factor * initial[j]
            rhs[3 * i] += p.sink_temperature_c * (1 - alpha)
            rhs[3 * i + 1] += (p.power_generation_w - p.bus_load_w) * step / 3600
            equalities[3 * i + 1, base + Column.SPILLED_ENERGY] = 1
            inequalities[2 * i + 1, minimum_energy] = 1
            inequalities[2 * i + 1, base + Column.BATTERY] = -1
            upper[2 * i] = step
            bounds[base + Column.TEMPERATURE] = (
                None,
                limits.max_propulsion_temperature_c - self.thermal_margin_c,
            )
            bounds[base + Column.BATTERY] = (
                limits.min_battery_energy_wh + 1,
                p.battery_capacity_wh,
            )
            h = limits.max_abs_wheel_momentum_nms - self.momentum_margin_nms
            # Keep both equal-performance thrusters exercised and leave torque
            # reserve in either direction until telemetry identifies an asymmetry.
            rates = tuple(self.effective_acceleration(mode) for mode in self.modes)
            if elapsed + step <= fine_duration + 1e-7 and (
                max(rates) - min(rates) < 0.05 * max(max(rates), 1e-12)
            ):
                h = min(h, max(0.1, abs(telemetry.wheel_momentum_nms)))
            bounds[base + Column.MOMENTUM] = self.momentum_bounds(h)
            bounds[base + Column.SPILLED_ENERGY] = (0, None)
            elapsed += step
        bounds[minimum_energy] = (
            limits.min_battery_energy_wh + 1,
            min(telemetry.battery_energy_wh, p.battery_capacity_wh),
        )
        rhs[-1] = self.brief.target_delta_v_mps - self.dv
        with warnings.catch_warnings():
            # SciPy forwards this native HiGHS option. Do not hide unrelated
            # numerical or configuration warnings from the optimizer.
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Unrecognized options detected: .*threads.*"
                    r"passed to HiGHS verbatim.*"
                ),
                category=OptimizeWarning,
            )
            result = linprog(
                objective,
                A_ub=inequalities.tocsr(),
                b_ub=upper,
                A_eq=equalities.tocsr(),
                b_eq=rhs,
                bounds=bounds,
                method="highs",
                options={
                    "threads": 1,
                    "time_limit": min(
                        1.0,
                        max(
                            0.01,
                            self.lp_cpu_limit
                            - (time.process_time() - self.cpu_started),
                        ),
                    ),
                },
            )
        self.lp_calls += 1
        if not result.success or not np.all(np.isfinite(result.x)):
            self.lp_failures += 1
            return None
        values: FloatArray = np.asarray(result.x, dtype=np.float64)
        return values, steps

    def prefer_beam(self) -> bool:
        """Select the numerical planner when estimated thrust is asymmetric."""
        rates = tuple(self.acceleration.values())
        return self.USE_BEAM_ON_ASYMMETRY and (
            max(rates) - min(rates) >= 0.05 * max(max(rates), 1e-12)
        )

    def effective_acceleration(self, mode: PropulsionMode) -> float:
        return self.actuator_terms(mode).acceleration_mps2

    def plan(self, telemetry: Telemetry, interval: float) -> list[PlanAction]:
        return self._plan_linear(telemetry, interval)

    def _plan_linear(self, telemetry: Telemetry, interval: float) -> list[PlanAction]:
        """Solve the global allocation, falling back within the same model."""
        if self.prefer_beam():
            return super().plan(telemetry, interval)
        p = self.brief.nominal_parameters
        if (
            not self.dump_context
            and self.brief.target_delta_v_mps - self.dv
            > max(self.acceleration.values()) * self.LP_TERMINAL_WINDOW_S
        ):
            return super().plan(telemetry, interval)
        if (
            time.process_time() - self.cpu_started > self.lp_cpu_limit
            or telemetry.propulsion_temperature_c < p.min_firing_temperature_c + 0.3
            or telemetry.propulsion_temperature_c
            > self.brief.limits.max_propulsion_temperature_c - self.thermal_margin_c
        ):
            return super().plan(telemetry, interval)
        deadline = min(
            self.brief.time_limit_s,
            max(
                self.REFERENCE_TIME_S,
                telemetry.time_s + interval,
                self.planned_end - 10,
            ),
        )
        solution = self.solve(telemetry, deadline)
        infeasible = deadline
        while solution is None and deadline < self.brief.time_limit_s:
            if time.process_time() - self.cpu_started > self.lp_cpu_limit:
                return super().plan(telemetry, interval)
            deadline = min(self.brief.time_limit_s, deadline + 120)
            solution = self.solve(telemetry, deadline)
        if solution is None:
            return super().plan(telemetry, interval)
        if deadline > infeasible:
            feasible = deadline
            for _ in range(4):
                middle = (infeasible + feasible) / 2
                candidate = self.solve(telemetry, middle)
                if candidate is None:
                    infeasible = middle
                else:
                    feasible, solution = middle, candidate
            deadline = feasible
        self.planned_end = deadline
        values, steps = solution
        state = PredictedState(
            self.dv,
            telemetry.propulsion_temperature_c,
            telemetry.battery_energy_wh,
            telemetry.wheel_momentum_nms,
        )
        actions = []
        offset = 0.0
        for i, step in enumerate(steps):
            if offset >= interval - 1e-7:
                break
            # Burn toward zero momentum first, preserving safety within each cell.
            order = sorted(
                self.modes,
                key=lambda m: abs(
                    state[3]
                    + firing_thruster(p, m).wheel_momentum_rate_nms_per_s * step
                ),
            )
            consumed = 0.0
            for mode in order:
                j = self.modes.index(mode)
                requested = max(0.0, values[VARIABLES_PER_STEP * i + j])
                burn = min(requested, self.maximum_burn(*state[1:], mode, requested))
                acceleration = self.effective_acceleration(mode)
                if acceleration > 1e-12:
                    burn = min(
                        burn, (self.brief.target_delta_v_mps - state[0]) / acceleration
                    )
                if burn > 1e-7:
                    actions.append(PlanAction(offset + consumed, mode, burn, burn))
                    state = self.predict(state, mode, burn, burn)
                    consumed += burn
            coast = max(0.0, step - consumed)
            if coast > 1e-7:
                actions.append(
                    PlanAction(offset + consumed, PropulsionMode.OFF, 0.0, coast)
                )
                state = self.predict(state, PropulsionMode.OFF, 0.0, coast)
            offset += step
        return actions
