"""Blue marble detection and conversion from image pixels to board XY.

The detector deliberately separates appearance from geometry:

* blue chroma proposes image regions;
* contour shape and expected projected size reject false positives;
* the calibrated camera ray is intersected with the plane through the marble
  centre (z = radius), expressed in the moving board frame.

Coordinates use the project convention: origin at the moving plate's
lower-left corner, +x right, +y up, and metres throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .board_pose import BoardPoseEstimator, PoseResult


@dataclass
class BallResult:
    pixel_xy: np.ndarray
    board_xy_m: np.ndarray
    radius_px: float
    confidence: float
    contour: np.ndarray


class BlueBallDetector:
    """Detect one blue marble inside the pose-derived board region."""

    def __init__(
        self,
        estimator: BoardPoseEstimator,
        *,
        radius_m: float = 0.0055,
        min_confidence: float = 0.54,
        global_recovery_interval: int = 5,
    ):
        if radius_m <= 0 or global_recovery_interval < 1:
            raise ValueError("ball radius must be positive")
        self.estimator = estimator
        self.geometry = estimator.geometry
        self.radius_m = float(radius_m)
        self.min_confidence = float(min_confidence)
        self.global_recovery_interval = int(global_recovery_interval)
        self._detection_calls = 0
        self.last_mask: np.ndarray | None = None

    # ---- calibrated geometry -------------------------------------------
    def _project(self, points_board: np.ndarray, pose: PoseResult) -> np.ndarray:
        points = np.asarray(points_board, dtype=np.float64).reshape(-1, 1, 3)
        rvec, _ = cv2.Rodrigues(pose.rotation)
        tvec = pose.translation.reshape(3, 1)
        if self.estimator.fisheye:
            projected, _ = cv2.fisheye.projectPoints(
                points, rvec, tvec, self.estimator.K, self.estimator.dist)
        else:
            projected, _ = cv2.projectPoints(
                points, rvec, tvec, self.estimator.K, self.estimator.dist)
        return projected.reshape(-1, 2)

    def board_mask(self, shape: tuple[int, int], pose: PoseResult) -> np.ndarray:
        """Return an image mask bounded by sampled, distortion-aware edges."""
        width = self.geometry.board_width_m
        height = self.geometry.board_height_m
        samples = 20
        bottom = np.column_stack((
            np.linspace(0.0, width, samples), np.zeros(samples)))
        right = np.column_stack((
            np.full(samples, width), np.linspace(0.0, height, samples)))
        top = np.column_stack((
            np.linspace(width, 0.0, samples), np.full(samples, height)))
        left = np.column_stack((
            np.zeros(samples), np.linspace(height, 0.0, samples)))
        xy = np.concatenate((bottom, right, top, left), axis=0)
        xyz = np.column_stack((xy, np.zeros(len(xy))))
        polygon = np.rint(self._project(xyz, pose)).astype(np.int32)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # The polygon is the playing-surface boundary at z=0, but the visible
        # ball silhouette extends roughly one radius beyond it when the marble
        # touches an outer rail. Clipping at the surface boundary cuts off half
        # the colour contour and biases fitEllipse inward. Expand only the
        # appearance ROI; pixel_to_board_xy still constrains the recovered
        # centre to the physical board (with one-radius tolerance).
        sample_xy = np.array([
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
            [0.5 * width, 0.5 * height],
        ])
        projected_radii = [
            self.expected_radius_px(point, pose) for point in sample_xy
        ]
        silhouette_margin_px = max(
            2, min(40, int(math.ceil(max(projected_radii))) + 2))
        diameter = 2 * silhouette_margin_px + 1
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
        )

        # Tag borders are black but appear blue under this camera's locked
        # white balance. They are known geometry, so remove their complete
        # mounting regions rather than asking the colour classifier to guess.
        for tag in self.geometry.tags.values():
            tag_polygon = self._project(tag.corners(), pose)
            centre = tag_polygon.mean(axis=0)
            expanded = centre + 1.55 * (tag_polygon - centre)
            cv2.fillConvexPoly(mask, np.rint(expanded).astype(np.int32), 0)
        return mask

    def pixel_to_board_xy(
        self, pixel_xy: np.ndarray, pose: PoseResult
    ) -> np.ndarray | None:
        """Intersect an undistorted camera ray with board z = ball radius."""
        undistorted = self.estimator.undistort(
            np.asarray(pixel_xy, dtype=np.float64).reshape(1, 2))[0]
        ray_camera = np.linalg.inv(self.estimator.K) @ np.array(
            [undistorted[0], undistorted[1], 1.0], dtype=np.float64)

        rotation_board_camera = pose.rotation.T
        camera_board = -rotation_board_camera @ pose.translation
        ray_board = rotation_board_camera @ ray_camera
        if abs(float(ray_board[2])) < 1e-9:
            return None
        distance = (self.radius_m - camera_board[2]) / ray_board[2]
        if distance <= 0:
            return None
        point = camera_board + distance * ray_board
        xy = point[:2]
        margin = self.radius_m
        if not (-margin <= xy[0] <= self.geometry.board_width_m + margin
                and -margin <= xy[1] <= self.geometry.board_height_m + margin):
            return None
        return xy.astype(np.float64)

    def expected_radius_px(self, board_xy_m: np.ndarray, pose: PoseResult) -> float:
        """Approximate projected radius at a candidate's board location."""
        x, y = (float(v) for v in board_xy_m)
        centre = np.array([[x, y, self.radius_m]], dtype=np.float64)
        offsets = np.array([
            [x + self.radius_m, y, self.radius_m],
            [x, y + self.radius_m, self.radius_m],
        ], dtype=np.float64)
        centre_px = self._project(centre, pose)[0]
        offset_px = self._project(offsets, pose)
        return float(np.mean(np.linalg.norm(offset_px - centre_px, axis=1)))

    # ---- appearance and candidate scoring ------------------------------
    @staticmethod
    def blue_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return broad and strict blue masks with brightness separated out."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        blue, green, red = cv2.split(frame)

        # The installed camera gives black board holes a weak blue cast, so a
        # brightness floor is essential in addition to hue. The broad mask
        # keeps shaded/highlighted parts of the actual blue marble.
        broad = (
            (hue >= 96) & (hue <= 116)
            & (saturation >= 90) & (blue >= 55)
            & (blue.astype(np.int16) >= green.astype(np.int16) + 15)
            & (blue.astype(np.int16) >= red.astype(np.int16) + 25)
        )
        # Strict mask supplies a confidence signal and rejects grey shadows.
        strict = (
            (hue >= 98) & (hue <= 112)
            & (saturation >= 120) & (blue >= 70)
            & (blue.astype(np.int16) >= green.astype(np.int16) + 25)
            & (blue.astype(np.int16) >= red.astype(np.int16) + 40)
        )
        return broad.astype(np.uint8) * 255, strict.astype(np.uint8) * 255

    def detect(
        self,
        frame: np.ndarray,
        pose: PoseResult,
        *,
        previous_xy_m: np.ndarray | None = None,
        max_step_m: float = 0.065,
    ) -> BallResult | None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("blue ball detector expects a BGR image")
        self._detection_calls += 1

        board_mask = self.board_mask(frame.shape[:2], pose)
        broad, strict = self.blue_masks(frame)
        broad = cv2.bitwise_and(broad, board_mask)
        strict = cv2.bitwise_and(strict, board_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        broad = cv2.morphologyEx(broad, cv2.MORPH_OPEN, kernel)
        broad = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, kernel)
        self.last_mask = broad

        contours, _ = cv2.findContours(
            broad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: BallResult | None = None
        best_selection_score = -math.inf
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if area < 12.0 or perimeter < 8.0:
                continue

            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                pixel = np.asarray(ellipse[0], dtype=np.float64)
                radius_px = 0.25 * float(ellipse[1][0] + ellipse[1][1])
            else:
                (cx, cy), radius_px = cv2.minEnclosingCircle(contour)
                pixel = np.array([cx, cy], dtype=np.float64)

            board_xy = self.pixel_to_board_xy(pixel, pose)
            if board_xy is None:
                continue
            distance_from_previous = 0.0
            if previous_xy_m is not None:
                distance_from_previous = float(np.linalg.norm(
                    board_xy - np.asarray(previous_xy_m, dtype=np.float64)))
                if distance_from_previous > max_step_m:
                    continue
            expected_radius = self.expected_radius_px(board_xy, pose)
            if expected_radius < 1.0:
                continue

            size_ratio = radius_px / expected_radius
            if not 0.45 <= size_ratio <= 2.2:
                continue
            size_score = math.exp(-1.8 * abs(math.log(size_ratio)))
            circularity = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))

            component_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, -1)
            component_pixels = max(1, cv2.countNonZero(component_mask))
            strict_fraction = cv2.countNonZero(
                cv2.bitwise_and(strict, component_mask)) / component_pixels
            confidence = float(
                0.45 * strict_fraction + 0.30 * circularity + 0.25 * size_score)

            result = BallResult(
                pixel_xy=pixel,
                board_xy_m=board_xy,
                radius_px=float(radius_px),
                confidence=confidence,
                contour=contour,
            )
            proximity_bonus = (
                0.0 if previous_xy_m is None else
                0.20 * math.exp(-0.5 * (distance_from_previous / 0.025) ** 2)
            )
            selection_score = confidence + proximity_bonus
            if best is None or selection_score > best_selection_score:
                best = result
                best_selection_score = selection_score

        # Keep the colour result as an identity anchor. A nearby Hough circle
        # can refine its centre from the physical edge even when highlights
        # leave only a small, off-centre blue core.
        colour_anchor = best

        # A ball touching a rail, wall, or handling tool can merge into one
        # large colour component and fail the contour-size test above. Recover
        # it from circular edge support, then require that the circle interior
        # is genuinely bright blue. The board crop keeps this affordable.
        nonzero = cv2.findNonZero(board_mask)
        run_hough = (colour_anchor is not None
                     or self._detection_calls % self.global_recovery_interval == 0)
        if nonzero is not None and run_hough:
            if colour_anchor is not None:
                anchor_expected_radius = self.expected_radius_px(
                    colour_anchor.board_xy_m, pose)
                half_size = int(math.ceil(max(
                    30.0,
                    3.0 * colour_anchor.radius_px,
                    2.5 * anchor_expected_radius,
                )))
                centre_x, centre_y = np.rint(
                    colour_anchor.pixel_xy).astype(int)
                x0 = max(0, centre_x - half_size)
                y0 = max(0, centre_y - half_size)
                x1 = min(frame.shape[1], centre_x + half_size + 1)
                y1 = min(frame.shape[0], centre_y + half_size + 1)
                crop_width, crop_height = x1 - x0, y1 - y0
            else:
                # No colour anchor: retain the expensive board-wide fallback
                # for recovery from a ball merged with another blue object.
                x0, y0, crop_width, crop_height = cv2.boundingRect(nonzero)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_crop = cv2.GaussianBlur(
                gray[y0:y0 + crop_height, x0:x0 + crop_width],
                (5, 5), 1.2)
            circles = cv2.HoughCircles(
                gray_crop,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(12, int(1.5 * (
                    colour_anchor.radius_px if colour_anchor is not None else 12))),
                param1=100,
                param2=16,
                minRadius=8,
                maxRadius=25,
            )
            if circles is not None:
                for crop_x, crop_y, radius_px in circles[0]:
                    pixel = np.array(
                        [crop_x + x0, crop_y + y0], dtype=np.float64)
                    board_xy = self.pixel_to_board_xy(pixel, pose)
                    if board_xy is None:
                        continue
                    distance_from_previous = 0.0
                    if previous_xy_m is not None:
                        distance_from_previous = float(np.linalg.norm(
                            board_xy - np.asarray(
                                previous_xy_m, dtype=np.float64)))
                        if distance_from_previous > max_step_m:
                            continue

                    expected_radius = self.expected_radius_px(board_xy, pose)
                    size_ratio = float(radius_px) / expected_radius
                    if not 0.55 <= size_ratio <= 2.0:
                        continue
                    anchored = (
                        colour_anchor is not None
                        and float(np.linalg.norm(
                            pixel - colour_anchor.pixel_xy))
                        <= max(4.0, 0.40 * expected_radius)
                    )
                    disk_x0 = max(0, int(math.floor(pixel[0] - radius_px)))
                    disk_y0 = max(0, int(math.floor(pixel[1] - radius_px)))
                    disk_x1 = min(frame.shape[1], int(math.ceil(
                        pixel[0] + radius_px)) + 1)
                    disk_y1 = min(frame.shape[0], int(math.ceil(
                        pixel[1] + radius_px)) + 1)
                    yy, xx = np.ogrid[disk_y0:disk_y1, disk_x0:disk_x1]
                    disk = ((xx - pixel[0]) ** 2 + (yy - pixel[1]) ** 2
                            <= float(radius_px) ** 2)
                    disk_pixels = max(1, int(np.count_nonzero(disk)))
                    broad_crop = broad[disk_y0:disk_y1, disk_x0:disk_x1]
                    strict_crop = strict[disk_y0:disk_y1, disk_x0:disk_x1]
                    broad_fraction = float(np.count_nonzero(
                        broad_crop[disk])) / disk_pixels
                    strict_fraction = float(np.count_nonzero(
                        strict_crop[disk])) / disk_pixels
                    if anchored:
                        # The colour candidate has already established that
                        # this is the marble. Its complete circular disk can
                        # contain white specular highlights and dark shading,
                        # so it need not be majority strict-blue.
                        if strict_fraction < 0.12 or broad_fraction < 0.30:
                            continue
                    elif strict_fraction < 0.60 or broad_fraction < 0.68:
                        continue
                    size_score = math.exp(-1.4 * abs(math.log(size_ratio)))
                    if anchored:
                        blue_support = (
                            0.55 * min(1.0, strict_fraction / 0.25)
                            + 0.45 * min(1.0, broad_fraction / 0.45)
                        )
                        confidence = float(
                            0.55 + 0.25 * blue_support
                            + 0.20 * size_score)
                    else:
                        confidence = float(
                            0.60 * strict_fraction
                            + 0.25 * broad_fraction
                            + 0.15 * size_score)

                    angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
                    contour = np.column_stack((
                        pixel[0] + float(radius_px) * np.cos(angles),
                        pixel[1] + float(radius_px) * np.sin(angles),
                    )).reshape(-1, 1, 2).astype(np.int32)
                    result = BallResult(
                        pixel_xy=pixel,
                        board_xy_m=board_xy,
                        radius_px=float(radius_px),
                        confidence=confidence,
                        contour=contour,
                    )
                    proximity_bonus = (
                        0.0 if previous_xy_m is None else
                        0.25 * math.exp(
                            -0.5 * (distance_from_previous / 0.025) ** 2)
                    )
                    selection_score = confidence + proximity_bonus
                    if best is None or selection_score > best_selection_score:
                        best = result
                        best_selection_score = selection_score

        if best is None or best.confidence < self.min_confidence:
            return None
        return best
