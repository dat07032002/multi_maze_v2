"""Build a metric maze occupancy map from a rectified camera image."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tag_vision.core.board_pose import BoardPoseEstimator, PoseResult


@dataclass
class CameraMazeMap:
    topdown_bgr: np.ndarray
    wall_mask: np.ndarray
    hole_mask: np.ndarray
    raw_occupied: np.ndarray
    occupied: np.ndarray
    clearance_mm: np.ndarray
    resolution_mm: float
    board_width_mm: float
    board_height_mm: float
    hole_centres_mm: list[tuple[float, float]]

    def board_to_grid(self, xy_mm) -> tuple[int, int]:
        x, y = (float(v) for v in xy_mm)
        col = int(round(x / self.resolution_mm))
        row = int(round((self.board_height_mm - y) / self.resolution_mm))
        return (min(max(row, 0), self.occupied.shape[0] - 1),
                min(max(col, 0), self.occupied.shape[1] - 1))

    def grid_to_board(self, row_col) -> tuple[float, float]:
        row, col = row_col
        return (float(col) * self.resolution_mm,
                self.board_height_mm - float(row) * self.resolution_mm)


def rectify_board(
    frame_bgr: np.ndarray,
    estimator: BoardPoseEstimator,
    pose: PoseResult,
    *,
    pixels_per_mm: float = 2.0,
) -> np.ndarray:
    """Warp the playing surface into lower-left-origin metric coordinates.

    The returned image is displayed conventionally (top row is board +Y), but
    every pixel has a fixed physical scale. Pose is evaluated for each frame,
    so small board motion between mapping frames does not blur the median map.
    """
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive")
    calibrated = cv2.resize(
        frame_bgr, estimator.image_size, interpolation=cv2.INTER_AREA)
    undistorted = estimator.undistort_frame(calibrated)
    width_m = estimator.geometry.board_width_m
    height_m = estimator.geometry.board_height_m
    corners_board = np.array([
        [0.0, 0.0, 0.0], [width_m, 0.0, 0.0],
        [width_m, height_m, 0.0], [0.0, height_m, 0.0],
    ], dtype=np.float64)
    source = estimator.project_undistorted(corners_board, pose).astype(np.float32)
    width_px = int(round(width_m * 1000.0 * pixels_per_mm))
    height_px = int(round(height_m * 1000.0 * pixels_per_mm))
    destination = np.array([
        [0, height_px - 1], [width_px - 1, height_px - 1],
        [width_px - 1, 0], [0, 0],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        undistorted, transform, (width_px, height_px),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _circular_holes(
    topdown_bgr: np.ndarray,
    pixels_per_mm: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    gray = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < 72).astype(np.uint8) * 255
    radius = max(1, int(round(0.8 * pixels_per_mm)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    centres: list[tuple[float, float]] = []
    height, width = gray.shape
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        area_mm2 = area_px / (pixels_per_mm ** 2)
        circularity = 4.0 * np.pi * area_px / (perimeter * perimeter)
        (cx, cy), radius_px = cv2.minEnclosingCircle(contour)
        radius_mm = radius_px / pixels_per_mm
        # Physical holes are nominally 7.5 mm radius. Broad limits tolerate
        # highlights, edge shadows, and the brown reflection across one hole.
        if not (80.0 <= area_mm2 <= 300.0
                and 5.0 <= radius_mm <= 11.0
                and circularity >= 0.48):
            continue
        x_mm = cx / pixels_per_mm
        y_mm = (height - 1 - cy) / pixels_per_mm
        # AprilTags live in the four corner pads and contain dark square/cross
        # components. Exclude only those pads, not nearby playable cells.
        corner_x = x_mm < 23.5 or x_mm > width / pixels_per_mm - 23.5
        corner_y = y_mm < 23.5 or y_mm > height / pixels_per_mm - 23.5
        if corner_x and corner_y:
            continue
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        centres.append((float(x_mm), float(y_mm)))
    centres.sort(key=lambda point: (-point[1], point[0]))
    return mask, centres


def build_camera_maze_map(
    topdown_bgr: np.ndarray,
    *,
    board_width_mm: float,
    board_height_mm: float,
    pixels_per_mm: float = 2.0,
    resolution_mm: float = 1.0,
    ball_radius_mm: float = 5.5,
    safety_margin_mm: float = 0.5,
) -> CameraMazeMap:
    """Segment walls/holes and return raw and configuration-space occupancy."""
    if resolution_mm <= 0 or ball_radius_mm <= 0 or safety_margin_mm < 0:
        raise ValueError("invalid map resolution, ball radius, or safety margin")
    hsv = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2HSV)
    # Yellow printed walls remain saturated even in shadow. These bounds avoid
    # the pale wood frame and the grey wall shadows.
    wall_high_res = cv2.inRange(
        hsv, np.array([17, 95, 85], np.uint8),
        np.array([42, 255, 255], np.uint8))
    close_px = max(1, int(round(0.8 * pixels_per_mm)))
    wall_high_res = cv2.morphologyEx(
        wall_high_res, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1)))
    hole_high_res, centres = _circular_holes(topdown_bgr, pixels_per_mm)

    width_cells = int(round(board_width_mm / resolution_mm))
    height_cells = int(round(board_height_mm / resolution_mm))

    def downsample(mask: np.ndarray) -> np.ndarray:
        reduced = cv2.resize(mask, (width_cells, height_cells),
                             interpolation=cv2.INTER_AREA)
        return reduced >= 64  # retain partially covered physical cells

    walls = downsample(wall_high_res)
    holes = downsample(hole_high_res)
    raw = walls | holes
    # The wooden frame contains the ball, but its inner edge is still an
    # obstacle. Mark one raw cell and let configuration-space inflation create
    # the correct centre-of-ball boundary.
    raw[[0, -1], :] = True
    raw[:, [0, -1]] = True
    inflation_cells = int(np.ceil(
        (ball_radius_mm + safety_margin_mm) / resolution_mm))
    diameter = 2 * inflation_cells + 1
    occupied = cv2.dilate(
        raw.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
    ).astype(bool)
    clearance = cv2.distanceTransform(
        (~occupied).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    clearance *= float(resolution_mm)
    return CameraMazeMap(
        topdown_bgr=topdown_bgr,
        wall_mask=walls,
        hole_mask=holes,
        raw_occupied=raw,
        occupied=occupied,
        clearance_mm=clearance,
        resolution_mm=float(resolution_mm),
        board_width_mm=float(board_width_mm),
        board_height_mm=float(board_height_mm),
        hole_centres_mm=centres,
    )
