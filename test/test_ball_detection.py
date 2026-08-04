"""Synthetic tests for blue-marble detection and board coordinates."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from tag_vision.core.ball_detection import BlueBallDetector
from tag_vision.core.board_geometry import BoardGeometry
from tag_vision.core.board_pose import BoardPoseEstimator, PoseResult, rotation_from_angles

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "calib" / "camera_calib.json"
GEOMETRY = ROOT / "calib" / "board_tags.json"


def synthetic_pose(geometry: BoardGeometry, alpha_deg=3.0, beta_deg=-2.0):
    centre = np.array([
        geometry.board_width_m / 2,
        geometry.board_height_m / 2,
        0.0,
    ])
    board_rotation = rotation_from_angles(
        math.radians(alpha_deg), math.radians(beta_deg))
    camera_flip = np.diag([1.0, -1.0, -1.0])
    rotation = camera_flip @ board_rotation
    translation = (
        camera_flip @ (-board_rotation @ centre)
        + np.array([0.0, 0.0, 0.32])
    )
    return PoseResult(
        alpha_rad=0.0,
        beta_rad=0.0,
        rotation=rotation,
        translation=translation,
        tag_ids=geometry.ids,
        reprojection_px=0.0,
        image_points=np.empty((0, 2)),
    )


class BlueBallDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = BoardGeometry.load(GEOMETRY)
        cls.estimator = BoardPoseEstimator(CALIB, cls.geometry)
        cls.detector = BlueBallDetector(cls.estimator, radius_m=0.0055)
        cls.pose = synthetic_pose(cls.geometry)

    def test_projected_centre_maps_back_to_board_xy(self):
        expected = np.array([0.142, 0.087])
        pixel = self.detector._project(
            [[expected[0], expected[1], self.detector.radius_m]], self.pose)[0]
        recovered = self.detector.pixel_to_board_xy(pixel, self.pose)
        np.testing.assert_allclose(recovered, expected, atol=1e-6)

    def test_detects_blue_ball_and_recovers_coordinates(self):
        expected = np.array([0.142, 0.087])
        width, height = self.estimator.image_size
        frame = np.full((height, width, 3), 190, dtype=np.uint8)
        pixel = self.detector._project(
            [[expected[0], expected[1], self.detector.radius_m]], self.pose)[0]
        radius_px = self.detector.expected_radius_px(expected, self.pose)
        cv2.circle(
            frame, tuple(np.rint(pixel).astype(int)), int(round(radius_px)),
            (170, 70, 5), -1, lineType=cv2.LINE_AA,
        )

        result = self.detector.detect(frame, self.pose)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.board_xy_m, expected, atol=0.001)
        self.assertGreater(result.confidence, 0.7)

    def test_rejects_non_blue_circle(self):
        width, height = self.estimator.image_size
        frame = np.full((height, width, 3), 190, dtype=np.uint8)
        cv2.circle(frame, (width // 2, height // 2), 8, (20, 20, 230), -1)
        self.assertIsNone(self.detector.detect(frame, self.pose))

    def test_previous_position_rejects_distant_blue_distractor(self):
        expected = np.array([0.060, 0.070])
        distractor = np.array([0.205, 0.165])
        width, height = self.estimator.image_size
        frame = np.full((height, width, 3), 190, dtype=np.uint8)
        for board_xy in (expected, distractor):
            pixel = self.detector._project(
                [[board_xy[0], board_xy[1], self.detector.radius_m]],
                self.pose,
            )[0]
            radius_px = self.detector.expected_radius_px(board_xy, self.pose)
            cv2.circle(
                frame, tuple(np.rint(pixel).astype(int)), int(round(radius_px)),
                (170, 70, 5), -1, lineType=cv2.LINE_AA,
            )

        result = self.detector.detect(
            frame, self.pose, previous_xy_m=expected)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.board_xy_m, expected, atol=0.001)

    def test_circle_fallback_recovers_ball_merged_with_blue_tool(self):
        expected = np.array([0.142, 0.087])
        width, height = self.estimator.image_size
        frame = np.full((height, width, 3), 190, dtype=np.uint8)
        pixel = self.detector._project(
            [[expected[0], expected[1], self.detector.radius_m]], self.pose)[0]
        radius_px = int(round(
            self.detector.expected_radius_px(expected, self.pose)))
        centre = tuple(np.rint(pixel).astype(int))
        colour = (170, 70, 5)
        cv2.circle(frame, centre, radius_px, colour, -1, lineType=cv2.LINE_AA)
        # A narrow blue arm touches the ball, creating one oversized colour
        # contour while leaving most of the circular edge visible.
        cv2.rectangle(
            frame,
            (centre[0] - radius_px // 3, centre[1] - 5 * radius_px),
            (centre[0] + radius_px // 3, centre[1] - radius_px + 2),
            colour, -1,
        )

        result = self.detector.detect(frame, self.pose)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.board_xy_m, expected, atol=0.003)

    def test_rejects_pixel_outside_board(self):
        self.assertIsNone(
            self.detector.pixel_to_board_xy(np.array([5.0, 5.0]), self.pose))


if __name__ == "__main__":
    unittest.main()
