"""LP execution, failure recovery and active thermal diagnosis checks."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from spacecraft_autonomy_challenge.challenge import problem
from spacecraft_autonomy_challenge.evaluation import evaluate
from spacecraft_autonomy_challenge.modifiers import Modifier
from spacecraft_autonomy_challenge.scenario import Scenario
from spacecraft_autonomy_challenge.simulation import run

from submission.autonomy.economic import EconomicStrategy
from submission.autonomy.linear import LPStrategy


class MoreHeat(Modifier):
    def modify_parameters(self, parameters, state, rng):
        return replace(
            parameters,
            thruster_1=replace(parameters.thruster_1, heat_w=48),
            thruster_2=replace(parameters.thruster_2, heat_w=48),
        )


class LPTests(unittest.TestCase):
    def test_fractional_completion(self):
        for controller in (LPStrategy, EconomicStrategy):
            with self.subTest(controller=controller.__name__):
                mission = replace(problem(), target_delta_v_mps=0.0023)
                metrics = evaluate(run(controller(), Scenario("tiny", mission)))
                self.assertTrue(metrics.safe)
                self.assertTrue(metrics.completion_declared)
                self.assertAlmostEqual(metrics.delta_v_mps, 0.0023, places=9)

    def test_solver_failure_uses_beam_fallback(self):
        mission = replace(problem(), target_delta_v_mps=0.03, time_limit_s=120)
        with patch.object(LPStrategy, "solve", return_value=None):
            metrics = evaluate(run(LPStrategy(), Scenario("fallback", mission)))
        self.assertTrue(metrics.safe)
        self.assertTrue(metrics.completion_declared)
        self.assertAlmostEqual(metrics.delta_v_mps, 0.03, places=9)

    def test_nonfinite_solver_result_uses_safe_fallback(self):
        mission = replace(problem(), target_delta_v_mps=0.03, time_limit_s=120)
        invalid = SimpleNamespace(success=True, x=np.array([np.nan]))
        with patch("submission.autonomy.linear.linprog", return_value=invalid):
            metrics = evaluate(run(LPStrategy(), Scenario("invalid-solver", mission)))
        self.assertTrue(metrics.safe)
        self.assertTrue(metrics.completion_declared)
        self.assertAlmostEqual(metrics.delta_v_mps, 0.03, places=9)

    def test_single_thruster_probes_identify_additional_heat(self):
        mission = replace(problem(), time_limit_s=180)
        controller = EconomicStrategy()
        result = run(controller, Scenario("more-heat", mission, (MoreHeat(),)))
        self.assertFalse(result.safety_violations)
        self.assertGreaterEqual(controller.diagnosis_count, 2)
        for heat in controller.heat.values():
            self.assertGreater(heat, 45)
            self.assertLess(heat, 50)


if __name__ == "__main__":
    unittest.main()
