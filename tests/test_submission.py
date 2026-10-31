"""Regression tests for the actual submitted conservative entry point."""

import unittest
from dataclasses import replace
from unittest.mock import Mock

from spacecraft_autonomy_challenge import Commander, PropulsionMode
from spacecraft_autonomy_challenge.challenge import problem, scenario, scenario_names
from spacecraft_autonomy_challenge.evaluation import evaluate
from spacecraft_autonomy_challenge.modifiers import TelemetryBias, ThrusterFlow
from spacecraft_autonomy_challenge.scenario import Scenario
from spacecraft_autonomy_challenge.simulation import run

from submission.autonomy.models import validate_plan
from submission.strategy import MyStrategy


class SubmissionTests(unittest.TestCase):
    def assert_safe_completion(self, case):
        metrics = evaluate(run(MyStrategy(), case))
        self.assertTrue(metrics.safe, metrics)
        self.assertTrue(metrics.completion_declared, metrics)
        self.assertLess(abs(metrics.delta_v_error_mps), 1e-6)
        return metrics

    def test_public_scenarios(self):
        for name in scenario_names():
            with self.subTest(scenario=name):
                self.assert_safe_completion(scenario(name))

    def test_temperature_sensor_underreads_by_two_degrees(self):
        metrics = self.assert_safe_completion(
            Scenario(
                "temperature-bias",
                problem(),
                (TelemetryBias("propulsion_temperature_c", -2, 3000),),
            )
        )
        self.assertLess(metrics.max_propulsion_temperature_c, 60)

    def test_zero_flow_recovers_using_healthy_thruster_and_dump(self):
        self.assert_safe_completion(
            Scenario(
                "zero-flow",
                problem(),
                (ThrusterFlow(PropulsionMode.THRUSTER_2, 0, 6000),),
            )
        )

    def test_cold_thrusters_are_heated_then_retried_without_assumed_completion(self):
        controller = MyStrategy()
        commander = Mock(spec=Commander, now_s=0.0)
        telemetry = scenario("nominal").brief().nominal_initial_telemetry
        controller.initialize(scenario("nominal").brief(), telemetry, commander)
        controller.acceleration = dict.fromkeys(controller.modes, 0.0)
        cold = replace(telemetry, propulsion_temperature_c=0.05, time_s=60)
        self.assertEqual(controller.plan(cold, 60)[0].mode, PropulsionMode.OFF)
        self.assertTrue(controller.used_heater)
        warm = replace(cold, propulsion_temperature_c=4)
        actions = controller.plan(warm, 60)
        validate_plan(actions, 60)
        self.assertFalse(controller.used_heater)
        self.assertGreater(actions[0].burn_s, 0)
        self.assertLessEqual(actions[0].burn_s, controller.RETRY_DURATION_S)
