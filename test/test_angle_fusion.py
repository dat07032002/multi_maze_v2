"""Hardware-free tests for camera/IMU complementary angle fusion."""
from __future__ import annotations

import math
import unittest

import numpy as np

from tag_vision.core.angle_fusion import CameraImuFusion


class AngleFusionTest(unittest.TestCase):
    def test_initializes_from_camera_absolute_reference(self):
        fusion = CameraImuFusion()
        got = fusion.update([1.0, -2.0], [1.2, -1.8], timestamp=0.0,
                            imu_timestamp=1)
        np.testing.assert_allclose(got.array, [1.0, -2.0])

    def test_imu_increment_carries_state_during_camera_dropout(self):
        fusion = CameraImuFusion()
        fusion.update([0.0, 0.0], [0.0, 0.0], timestamp=0.0,
                      imu_timestamp=1)
        got = fusion.update(None, [1.25, -0.5], timestamp=0.01,
                            imu_timestamp=2)
        np.testing.assert_allclose(got.array, [1.25, -0.5])
        self.assertTrue(got.imu_used)

    def test_same_imu_sample_is_not_integrated_twice(self):
        fusion = CameraImuFusion()
        fusion.update([0.0, 0.0], [0.0, 0.0], timestamp=0.0,
                      imu_timestamp=10)
        fusion.update(None, [1.0, 0.0], timestamp=0.01, imu_timestamp=11)
        got = fusion.update(None, [1.0, 0.0], timestamp=0.02,
                            imu_timestamp=11)
        self.assertAlmostEqual(got.alpha_deg, 1.0)
        self.assertFalse(got.imu_used)

    def test_camera_removes_imu_drift_at_configured_rate(self):
        fusion = CameraImuFusion(camera_time_constant_s=0.5)
        fusion.update([0.0, 0.0], [0.0, 0.0], timestamp=0.0,
                      imu_timestamp=1)
        fusion.update(None, [1.0, 0.0], timestamp=0.01, imu_timestamp=2)
        got = fusion.update([0.0, 0.0], [1.0, 0.0], timestamp=0.5,
                            imu_timestamp=2)
        self.assertAlmostEqual(got.alpha_deg, math.exp(-0.49 / 0.5), places=6)

    def test_large_camera_outlier_is_rejected_per_axis(self):
        fusion = CameraImuFusion(max_camera_residual_deg=2.0)
        fusion.update([0.0, 0.0], [0.0, 0.0], timestamp=0.0,
                      imu_timestamp=1)
        got = fusion.update([20.0, 1.0], [0.0, 0.0], timestamp=0.5,
                            imu_timestamp=1)
        self.assertAlmostEqual(got.alpha_deg, 0.0)
        self.assertGreater(got.beta_deg, 0.0)
        self.assertEqual(got.camera_used, (False, True))

    def test_reset_after_joint_zero(self):
        fusion = CameraImuFusion()
        fusion.update([3.0, 2.0], [3.0, 2.0], timestamp=0.0,
                      imu_timestamp=1)
        fusion.reset([0.0, 0.0], [0.0, 0.0], timestamp=1.0,
                     imu_timestamp=2)
        np.testing.assert_allclose(fusion.angles_deg, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
