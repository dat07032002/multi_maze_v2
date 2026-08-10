from __future__ import annotations

import unittest

from tag_vision.control.directional_calibration import (
    DirectionalAxis,
    DirectionalBranch,
    DirectionalMotorCalibration,
    DirectionalMotorOrigin,
)


class DirectionalCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.axis = DirectionalAxis(
            servo_id=2, level_counts=2000, min_counts=1000, max_counts=3000,
            up=DirectionalBranch((-2.0, 0.0, 2.0), (1600, 2050, 2400)),
            down=DirectionalBranch((-2.0, 0.0, 2.0), (1500, 1900, 2300)))

    def test_interpolates_selected_branch(self):
        self.assertEqual(self.axis.counts_for(1.0, 1), 2225)
        self.assertEqual(self.axis.counts_for(1.0, -1), 2100)

    def test_extrapolates_then_clamps(self):
        self.assertEqual(self.axis.counts_for(20.0, 1), 3000)

    def test_origin_tracks_logical_direction(self):
        other = DirectionalAxis(
            servo_id=1, level_counts=2100, min_counts=1000, max_counts=3000,
            up=DirectionalBranch((-2.0, 2.0), (2500, 1700)),
            down=DirectionalBranch((-2.0, 2.0), (2600, 1800)))
        calibration = DirectionalMotorCalibration(
            alpha=self.axis, beta=other, source_run="test")
        origin = DirectionalMotorOrigin(calibration)
        self.assertEqual(origin.targets(1.0, 0.0)[2], 2225)
        self.assertEqual(origin.targets(0.0, 0.0)[2], 1900)


if __name__ == "__main__":
    unittest.main()
