"""Tests for the board pose path.

The important one is test_recovers_synthetic_tilt: it renders tags at a known
board tilt through the real calibration, runs the full detect-undistort-solve
path, and checks the recovered angles match what was rendered. That is the only
check here that would catch a wrong angle convention, a wrong corner ordering,
or a sign flip -- all of which produce plausible-looking output otherwise.
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from tag_vision.core.board_geometry import BoardGeometry, Tag, template
from tag_vision.core.board_pose import (
    BoardPoseEstimator,
    angles_from_rotation,
    rotation_from_angles,
)

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "calib" / "camera_calib.json"
GEOMETRY = ROOT / "calib" / "board_tags.json"


class AngleConventionTest(unittest.TestCase):
    def test_round_trips(self):
        for alpha_deg in (-12, -5, 0, 3, 11):
            for beta_deg in (-12, -7, 0, 4, 12):
                alpha, beta = math.radians(alpha_deg), math.radians(beta_deg)
                out = angles_from_rotation(rotation_from_angles(alpha, beta))
                self.assertAlmostEqual(out[0], alpha, places=9)
                self.assertAlmostEqual(out[1], beta, places=9)

    def test_matches_simulator_hinge_order(self):
        """R must be Rx(alpha) @ Ry(beta), the MJCF tilt_x -> tilt_y order."""
        alpha, beta = math.radians(7.0), math.radians(-4.0)
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        np.testing.assert_allclose(
            rotation_from_angles(alpha, beta), rx @ ry, atol=1e-12)

    def test_pure_x_tilt_leaves_beta_zero(self):
        alpha, beta = angles_from_rotation(rotation_from_angles(0.15, 0.0))
        self.assertAlmostEqual(alpha, 0.15, places=9)
        self.assertAlmostEqual(beta, 0.0, places=9)

    def test_pure_y_tilt_leaves_alpha_zero(self):
        alpha, beta = angles_from_rotation(rotation_from_angles(0.0, -0.12))
        self.assertAlmostEqual(alpha, 0.0, places=9)
        self.assertAlmostEqual(beta, -0.12, places=9)


class GeometryTest(unittest.TestCase):
    def test_template_loads_and_validates(self):
        geometry = BoardGeometry.load(GEOMETRY)
        self.assertEqual(len(geometry.tags), 4)
        # The nominal template is coplanar, so exactly that warning is expected.
        warnings = geometry.validate()
        self.assertTrue(any("coplanar" in w for w in warnings))
        self.assertFalse(any("outside the board" in w for w in warnings))

    def test_corner_order_matches_aruco(self):
        tag = Tag(tag_id=0, x_m=0.010, y_m=0.020, size_m=0.018)
        corners = tag.corners()
        # aruco order: top-left, top-right, bottom-right, bottom-left
        np.testing.assert_allclose(corners[0], [0.010, 0.038, 0.0], atol=1e-9)
        np.testing.assert_allclose(corners[1], [0.028, 0.038, 0.0], atol=1e-9)
        np.testing.assert_allclose(corners[2], [0.028, 0.020, 0.0], atol=1e-9)
        np.testing.assert_allclose(corners[3], [0.010, 0.020, 0.0], atol=1e-9)

    def test_rotated_tag_preserves_aruco_corner_correspondence(self):
        tag = Tag(
            tag_id=1, x_m=0.010, y_m=0.020, size_m=0.018,
            rotation_deg=-90.0,
        )
        np.testing.assert_allclose(
            tag.corners(),
            [[0.028, 0.038, 0.0], [0.028, 0.020, 0.0],
             [0.010, 0.020, 0.0], [0.010, 0.038, 0.0]],
            atol=1e-9,
        )

    def test_rejects_too_few_tags(self):
        with self.assertRaises(ValueError):
            BoardGeometry([Tag(0, 0.0, 0.0, 0.018), Tag(1, 0.1, 0.0, 0.018)])

    def test_flags_a_tag_off_the_board(self):
        geometry = BoardGeometry([
            Tag(0, 0.006, 0.006, 0.018),
            Tag(1, 0.235, 0.006, 0.018),
            Tag(2, 0.400, 0.205, 0.018),      # off the right edge
            Tag(3, 0.006, 0.205, 0.018),
        ])
        self.assertTrue(any("outside the board" in w for w in geometry.validate()))

    def test_riser_makes_it_non_planar(self):
        geometry = BoardGeometry([
            Tag(0, 0.006, 0.006, 0.018, z_m=0.000),
            Tag(1, 0.235, 0.006, 0.018, z_m=0.009),
            Tag(2, 0.235, 0.205, 0.018, z_m=0.000),
            Tag(3, 0.006, 0.205, 0.018, z_m=0.009),
        ])
        self.assertFalse(geometry.is_planar())
        self.assertFalse(any("coplanar" in w for w in geometry.validate()))


def render_board(estimator, geometry, alpha, beta, distance_m=0.32):
    """Project the tags at a known tilt and draw them into a synthetic frame."""
    width, height = estimator.image_size
    image = np.full((height, width), 255, np.uint8)

    board_centre = np.array(
        [geometry.board_width_m / 2, geometry.board_height_m / 2, 0.0])
    rotation = rotation_from_angles(alpha, beta)
    # Camera above the board centre looking down: board +z toward the camera,
    # so flip x and z to build a camera frame with +z into the scene.
    flip = np.diag([1.0, -1.0, -1.0])
    R_cam_board = flip @ rotation
    t = flip @ (-rotation @ board_centre) + np.array([0.0, 0.0, distance_m])
    rvec, _ = cv2.Rodrigues(R_cam_board)

    for tag_id in geometry.ids:
        tag = geometry.tags[tag_id]
        corners3d = tag.corners().astype(np.float64)
        if estimator.fisheye:
            projected, _ = cv2.fisheye.projectPoints(
                corners3d.reshape(-1, 1, 3), rvec, t.reshape(3, 1),
                estimator.K, estimator.dist)
        else:
            projected, _ = cv2.projectPoints(
                corners3d, rvec, t, estimator.K, estimator.dist)
        pts = projected.reshape(-1, 2)

        # Warp the tag bitmap onto its projected quad.
        marker = cv2.aruco.generateImageMarker(estimator.dictionary, tag_id, 80)
        source = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], np.float32)
        # marker row 0 is the tag top -> matches corner order TL,TR,BR,BL
        homography = cv2.getPerspectiveTransform(source, pts.astype(np.float32))
        warped = cv2.warpPerspective(
            marker, homography, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT,
            dst=image.copy())
        mask = cv2.warpPerspective(
            np.full((80, 80), 255, np.uint8), homography, (width, height),
            flags=cv2.INTER_NEAREST)
        image[mask > 0] = warped[mask > 0]
    return image, R_cam_board


class SyntheticPoseTest(unittest.TestCase):
    """The test that would catch a wrong convention or corner ordering."""

    @classmethod
    def setUpClass(cls):
        cls.geometry = BoardGeometry.load(GEOMETRY)
        cls.estimator = BoardPoseEstimator(CALIB, cls.geometry)

    def test_recovers_synthetic_tilt(self):
        cases = [(0.0, 0.0), (6.0, 0.0), (0.0, -5.0), (-4.0, 3.0), (9.0, 8.0)]
        for alpha_deg, beta_deg in cases:
            alpha, beta = math.radians(alpha_deg), math.radians(beta_deg)
            image, R_true = render_board(
                self.estimator, self.geometry, alpha, beta)

            result = self.estimator.estimate(image)
            self.assertIsNotNone(
                result, f"no pose recovered at ({alpha_deg}, {beta_deg}) deg")
            self.assertEqual(len(result.tag_ids), 4)
            self.assertLess(result.reprojection_px, 2.0)

            # Compare against the truth expressed the same way the estimator
            # does: undo the camera flip to get the plate rotation back.
            flip = np.diag([1.0, -1.0, -1.0])
            recovered = angles_from_rotation(flip @ result.rotation)
            self.assertAlmostEqual(
                math.degrees(recovered[0]), alpha_deg, delta=0.5,
                msg=f"alpha at ({alpha_deg}, {beta_deg})")
            self.assertAlmostEqual(
                math.degrees(recovered[1]), beta_deg, delta=0.5,
                msg=f"beta at ({alpha_deg}, {beta_deg})")

    def test_zeroing_makes_a_tilted_reference_read_zero(self):
        alpha, beta = math.radians(3.0), math.radians(-2.0)
        image, _ = render_board(self.estimator, self.geometry, alpha, beta)
        result = self.estimator.estimate(image)
        self.assertIsNotNone(result)

        self.estimator.set_zero(result.rotation)
        try:
            zeroed = self.estimator.estimate(image)
            self.assertAlmostEqual(zeroed.alpha_deg, 0.0, delta=0.05)
            self.assertAlmostEqual(zeroed.beta_deg, 0.0, delta=0.05)
        finally:
            self.estimator.clear_zero()

    def test_ignores_unknown_tag_ids(self):
        image, _ = render_board(self.estimator, self.geometry, 0.0, 0.0)
        found = self.estimator.detect(image)
        self.assertEqual(set(found), set(self.geometry.ids))

    def test_undistorted_projection_matches_undistorted_pixels(self):
        image, _ = render_board(self.estimator, self.geometry, 0.0, 0.0)
        result = self.estimator.estimate(image)
        self.assertIsNotNone(result)
        points = np.array([
            [0.0, 0.0, 0.0],
            [self.geometry.board_width_m, 0.0, 0.0],
            [0.0, self.geometry.board_height_m, 0.0],
        ])
        rvec, _ = cv2.Rodrigues(result.rotation)
        distorted, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3), rvec,
            result.translation.reshape(3, 1), self.estimator.K,
            self.estimator.dist,
        )
        expected = self.estimator.undistort(distorted.reshape(-1, 2))
        actual = self.estimator.project_undistorted(points, result)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_undistorted_view_keeps_calibrated_frame_size(self):
        width, height = self.estimator.image_size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        rectified = self.estimator.undistort_frame(frame)
        self.assertEqual(rectified.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
