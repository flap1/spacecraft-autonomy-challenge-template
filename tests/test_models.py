"""Physical identities and validation at the optimizer/executor boundary."""

import math
import unittest
from dataclasses import replace

from spacecraft_autonomy_challenge import PropulsionMode
from spacecraft_autonomy_challenge.challenge import problem

from submission.autonomy.models import DUMP, PlanAction, validate_plan
from submission.autonomy.thermal import heat_response, temperature_after


class ThermalTests(unittest.TestCase):
    def test_timed_pulse_matches_two_stage_thermal_solution(self):
        parameters = problem().parameters
        explicit = temperature_after(
            parameters, temperature_after(parameters, 20, 40, 30), 0, 30
        )
        weighted = temperature_after(parameters, 20, 0, 60) + 40 * heat_response(
            parameters, 0, 30, 60
        )
        self.assertAlmostEqual(explicit, weighted, places=12)

    def test_zero_conductance_has_no_division_by_zero(self):
        parameters = replace(problem().parameters, thermal_conductance_w_per_k=0)
        self.assertEqual(temperature_after(parameters, 20, 40, 60), 21.2)
        self.assertEqual(heat_response(parameters, 10, 20, 60), 0.01)


class PlanValidationTests(unittest.TestCase):
    def test_mixed_firing_dump_and_coast_plan(self):
        validate_plan(
            [
                PlanAction(0, PropulsionMode.THRUSTER_1, 10, 10),
                PlanAction(10, DUMP, 5, 5),
                PlanAction(15, PropulsionMode.OFF, 0, 45),
            ],
            60,
        )

    def test_invalid_solver_times_are_rejected(self):
        invalid_plans = (
            [PlanAction(math.nan, PropulsionMode.OFF, 0, 5)],
            [PlanAction(0, PropulsionMode.THRUSTER_1, 6, 5)],
            [PlanAction(-0.01, PropulsionMode.OFF, 0, 5)],
            [PlanAction(59, PropulsionMode.OFF, 0, 5)],
            [PlanAction(0, DUMP, 5, 5), PlanAction(4, PropulsionMode.OFF, 0, 5)],
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                validate_plan(plan, 60)
