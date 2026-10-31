"""End-to-end controller checks using ground truth only in the test harness."""

import unittest
from dataclasses import replace

from spacecraft_autonomy_challenge import PropulsionMode
from spacecraft_autonomy_challenge.challenge import problem, scenario, scenario_names
from spacecraft_autonomy_challenge.evaluation import evaluate
from spacecraft_autonomy_challenge.modifiers import TelemetryBias, ThrusterEfficiency
from spacecraft_autonomy_challenge.scenario import Scenario
from spacecraft_autonomy_challenge.simulation import run

from submission.autonomy.beam import BeamStrategy as MyStrategy


class StrategyTests(unittest.TestCase):
    def assert_success(self, case):
        result = run(MyStrategy(), case)
        metrics = evaluate(result)
        self.assertTrue(metrics.safe, metrics)
        self.assertTrue(metrics.completion_declared, result.termination)
        self.assertLess(abs(metrics.delta_v_error_mps), 1e-6, metrics)
        return result

    def test_public_scenarios(self):
        for name in scenario_names():
            with self.subTest(name=name):
                self.assert_success(scenario(name))

    def test_combined_unaligned_sensor_jump_and_degradation(self):
        self.assert_success(
            Scenario(
                name="combined",
                problem=problem(),
                modifiers=(
                    ThrusterEfficiency(PropulsionMode.THRUSTER_2, 0.7, 3637.0),
                    TelemetryBias("delta_v_mps", -0.4, 8003.0),
                ),
            )
        )

    def test_hot_start_requires_partial_burns(self):
        mission = problem()
        mission = replace(
            mission,
            initial_state=replace(mission.initial_state, propulsion_temperature_c=59.0),
        )
        result = self.assert_success(Scenario(name="hot-start", problem=mission))
        self.assertTrue(
            any(
                abs(command.time_s / 60 - round(command.time_s / 60)) > 1e-6
                for command in result.executed_commands[:-2]
            )
        )

    def test_multiple_thrusters_within_one_observation_interval(self):
        mission = replace(problem(), time_limit_s=120.0)
        controller = MyStrategy()
        result = run(controller, Scenario(name="short", problem=mission))
        switches = [
            command
            for command in result.executed_commands
            if 0 < command.time_s < 60 and hasattr(command.command, "mode")
        ]
        self.assertGreater(len(switches), 1)
        self.assertEqual(
            {command.command.mode for command in switches},
            {PropulsionMode.THRUSTER_1, PropulsionMode.THRUSTER_2},
        )
        # Net wheel momentum nearly cancels, but neither thrust estimate is zero.
        for acceleration in controller.acceleration.values():
            self.assertAlmostEqual(acceleration, 0.001, places=10)

    def test_completion_at_fractional_second_inside_fine_step(self):
        mission = replace(problem(), target_delta_v_mps=0.0023)
        result = self.assert_success(Scenario(name="tiny-target", problem=mission))
        self.assertAlmostEqual(result.termination.time_s, 2.3, places=8)

    def test_low_battery_recharges_before_long_burn(self):
        mission = problem()
        mission = replace(
            mission,
            initial_state=replace(mission.initial_state, battery_energy_wh=12.0),
        )
        self.assert_success(Scenario(name="low-battery", problem=mission))


if __name__ == "__main__":
    unittest.main()
