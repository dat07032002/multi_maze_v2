from __future__ import annotations

import unittest

from tag_vision.control.directional_calibration import (
    DirectionalAxis, DirectionalBranch, DirectionalMotorCalibration)
from tag_vision.control.fused_trim import FusedStaticTrim


def calibration():
    alpha = DirectionalAxis(2, 2000, 1000, 3000,
                            DirectionalBranch((-1, 1), (1800, 2200)),
                            DirectionalBranch((-1, 1), (1750, 2150)))
    beta = DirectionalAxis(1, 2100, 1000, 3000,
                           DirectionalBranch((-1, 1), (2300, 1900)),
                           DirectionalBranch((-1, 1), (2350, 1950)))
    return DirectionalMotorCalibration(
        alpha, beta, "test", ((0.005, 0.001), (0.0005, -0.005)))


class FusedTrimTest(unittest.TestCase):
    def test_waits_until_settled_and_delayed(self):
        trim = FusedStaticTrim(calibration(), settle_delay_s=1.0)
        trim.arm((1, 1), {1: 2100, 2: 2000}, 0.0)
        self.assertIsNone(trim.update((0, 0), timestamp=2.0, settled=False))
        self.assertIsNone(trim.update((0, 0), timestamp=0.5, settled=True))

    def test_uses_coupled_inverse_and_servo_ids(self):
        trim = FusedStaticTrim(calibration(), gain=1.0,
                               max_step_counts=1000, min_step_counts=0)
        trim.arm((1, 1), {1: 2100, 2: 2000}, 0.0)
        got = trim.update((0, 0), timestamp=1.0, settled=True)
        # J @ [alpha-servo delta, beta-servo delta] == [1, 1].
        self.assertEqual(got.delta_counts, (235, -176))
        self.assertEqual(got.counts, {1: 1924, 2: 2235})

    def test_converges_inside_tolerance_without_moving(self):
        trim = FusedStaticTrim(calibration(), tolerance_deg=0.1,
                               convergence_hold_s=0.2)
        trim.arm((1, -1), {1: 2100, 2: 2000}, 0.0)
        self.assertIsNone(trim.update(
            (0.95, -1.04), timestamp=1.0, settled=True))
        got = trim.update((0.95, -1.04), timestamp=1.21, settled=True)
        self.assertTrue(got.converged)
        self.assertFalse(trim.active)

    def test_leaving_tolerance_resets_convergence_hold(self):
        trim = FusedStaticTrim(calibration(), tolerance_deg=0.1,
                               convergence_hold_s=0.4)
        trim.arm((0, 0), {1: 2100, 2: 2000}, 0.0)
        self.assertIsNone(trim.update((0.05, 0), timestamp=1.0, settled=True))
        moved = trim.update((0.2, 0), timestamp=1.2, settled=True)
        self.assertFalse(moved.converged)

    def test_caps_each_step_and_iterations(self):
        trim = FusedStaticTrim(calibration(), max_step_counts=30,
                               max_iterations=1)
        trim.arm((4, 4), {1: 2100, 2: 2000}, 0.0)
        first = trim.update((0, 0), timestamp=1.0, settled=True)
        self.assertLessEqual(max(map(abs, first.delta_counts)), 30)
        final = trim.update((0, 0), timestamp=2.0, settled=True)
        self.assertTrue(final.exhausted)

    def test_does_not_nudge_axis_already_inside_tolerance(self):
        trim = FusedStaticTrim(calibration(), tolerance_deg=0.1,
                               gain=1.0, min_step_counts=20)
        trim.arm((0, 1), {1: 2100, 2: 2000}, 0.0)
        got = trim.update((-0.05, 0), timestamp=1.0, settled=True)
        self.assertEqual(got.delta_counts[0], 0)
        self.assertNotEqual(got.delta_counts[1], 0)


if __name__ == "__main__":
    unittest.main()
