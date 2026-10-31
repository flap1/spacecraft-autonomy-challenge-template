"""Check hybrid actuator planning and adaptation against ground-truth outcomes."""

import unittest
from dataclasses import replace

from spacecraft_autonomy_challenge import PropulsionMode
from spacecraft_autonomy_challenge.challenge import problem
from spacecraft_autonomy_challenge.evaluation import evaluate
from spacecraft_autonomy_challenge.modifiers import Modifier, ThrusterEfficiency
from spacecraft_autonomy_challenge.scenario import Scenario
from spacecraft_autonomy_challenge.simulation import run

from submission.autonomy.dump import DumpEconomicStrategy


class LateHeatJump(Modifier):
    def modify_parameters(self, parameters, state, rng):
        if state.time_s < 121:
            return parameters
        return replace(
            parameters,
            thruster_1=replace(parameters.thruster_1, heat_w=48),
            thruster_2=replace(parameters.thruster_2, heat_w=48),
        )


class EconomicTests(unittest.TestCase):
    def test_strong_failure_uses_dump_and_beats_weak_thruster_recovery(self):
        case = Scenario(
            "severe",
            problem(),
            (ThrusterEfficiency(PropulsionMode.THRUSTER_1, 0.35, 4877),),
        )
        controller = DumpEconomicStrategy()
        metrics = evaluate(run(controller, case))
        self.assertTrue(metrics.safe)
        self.assertTrue(metrics.completion_declared)
        self.assertLess(abs(metrics.delta_v_error_mps), 1e-6)
        self.assertGreater(controller.dump_intervals, 0)
        # Baseline needs about 13485 s on this case. This is a physical benefit,
        # rather than a check that the implementation echoes its own prediction.
        self.assertLess(metrics.end_time_s, 12000)
        self.assertAlmostEqual(controller.net_generation_w, 20, places=5)

    def test_unobserved_twenty_percent_heat_jump_near_thermal_boundary(self):
        mission = problem()
        mission = replace(
            mission,
            target_delta_v_mps=0.6,
            time_limit_s=1800,
            initial_state=replace(mission.initial_state, propulsion_temperature_c=59),
        )
        controller = DumpEconomicStrategy()
        metrics = evaluate(
            run(controller, Scenario("heat", mission, (LateHeatJump(),)))
        )
        self.assertTrue(metrics.safe)
        self.assertTrue(metrics.completion_declared)
        self.assertGreater(controller.reset_count, 0)
        self.assertLess(abs(metrics.delta_v_error_mps), 1e-6)

    def test_short_symmetric_run_does_not_dump(self):
        mission = replace(problem(), target_delta_v_mps=0.05, time_limit_s=120)
        controller = DumpEconomicStrategy()
        metrics = evaluate(run(controller, Scenario("short", mission)))
        self.assertTrue(metrics.safe)
        self.assertTrue(metrics.completion_declared)
        self.assertEqual(controller.dump_intervals, 0)
        self.assertAlmostEqual(metrics.delta_v_mps, 0.05, places=9)


if __name__ == "__main__":
    unittest.main()
