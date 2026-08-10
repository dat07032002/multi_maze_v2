"""Pure tests for keyboard motor command mapping; no hardware is opened."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from contract.servo_contract import ServoContract
from tools.keyboard_motor_compare import MotorOrigin, TerminalKeys, key_action

ROOT = Path(__file__).resolve().parents[1]


class KeyboardMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = ServoContract.from_json(
            ROOT / "calib" / "servo_calibration.json")

    def test_ascii_and_linux_arrow_keys(self):
        self.assertEqual(key_action(ord("a")), "alpha_down")
        self.assertEqual(key_action(ord("w")), "beta_up")
        self.assertEqual(key_action(65361), "alpha_down")
        self.assertEqual(key_action(65362), "beta_up")

    def test_current_counts_are_zero_angle_origin(self):
        base = {1: 2200, 2: 2075}
        origin = MotorOrigin(self.contract, base)
        self.assertEqual(origin.targets(0.0, 0.0), base)

    def test_terminal_reader_is_disabled_without_tty(self):
        # pytest stdin is not an interactive terminal. Construction and a
        # non-blocking poll must still be harmless for headless validation.
        keys = TerminalKeys()
        if keys.fd is None:
            self.assertEqual(keys.poll(), -1)

    def test_angle_step_uses_measured_gain_and_sign(self):
        base = {1: 2200, 2: 2075}
        origin = MotorOrigin(self.contract, base)
        got = origin.targets(1.0, -1.0)
        expected_roll = round(
            base[self.contract.roll.servo_id] + self.contract.roll.sign
            * self.contract.roll.counts_per_rad * math.radians(1.0))
        expected_pitch = round(
            base[self.contract.pitch.servo_id] + self.contract.pitch.sign
            * self.contract.pitch.counts_per_rad * math.radians(-1.0))
        self.assertEqual(got, {
            self.contract.roll.servo_id: expected_roll,
            self.contract.pitch.servo_id: expected_pitch,
        })


if __name__ == "__main__":
    unittest.main()
