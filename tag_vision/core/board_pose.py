"""Board pose from four AprilTags on the tilting plate.

Angle convention is taken from the simulator, not invented here. In the MJCF
the plate hangs off two hinges:

    <joint name="tilt_x" axis="1 0 0">      parent
      <joint name="tilt_y" axis="0 1 0">    child

and `TagMazeSim.board_angles()` returns `[qpos(tilt_x), qpos(tilt_y)]`. So the
board rotation is R = Rx(alpha) @ Ry(beta), giving

    R = [[ cb,      0,    sb   ],
         [ sa*sb,  ca,  -sa*cb ],
         [-ca*sb,  sa,   ca*cb ]]

from which alpha = atan2(R[2,1], R[1,1]) and beta = atan2(R[0,2], R[0,0]).
The atan2 forms are used rather than asin because they stay well conditioned
and keep the correct quadrant; the previous estimator mixed asin and a bare
arctan, which differ by 180 degrees whenever R[2,2] < 0.

Tags sit only on the moving plate, so solvePnP gives plate-relative-to-CAMERA.
Board tilt is that measured against a zero captured with the board level, which
means the zero is only valid while the camera does not move. Adding one tag on
the fixed frame would remove that dependence.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .board_geometry import BoardGeometry

APRILTAG_FAMILIES = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
}


def rotation_from_angles(alpha: float, beta: float) -> np.ndarray:
    """R = Rx(alpha) @ Ry(beta), matching the simulator's hinge order."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    return np.array(
        [
            [cb, 0.0, sb],
            [sa * sb, ca, -sa * cb],
            [-ca * sb, sa, ca * cb],
        ],
        dtype=np.float64,
    )


def angles_from_rotation(rotation: np.ndarray) -> tuple[float, float]:
    """Inverse of rotation_from_angles. Returns (alpha, beta) in radians."""
    r = np.asarray(rotation, dtype=np.float64)
    alpha = math.atan2(r[2, 1], r[1, 1])
    beta = math.atan2(r[0, 2], r[0, 0])
    return alpha, beta


@dataclass
class PoseResult:
    alpha_rad: float
    beta_rad: float
    rotation: np.ndarray          # plate -> camera
    translation: np.ndarray       # plate origin in camera frame, metres
    tag_ids: tuple[int, ...]
    reprojection_px: float
    image_points: np.ndarray      # undistorted, as used by the solve

    @property
    def alpha_deg(self) -> float:
        return math.degrees(self.alpha_rad)

    @property
    def beta_deg(self) -> float:
        return math.degrees(self.beta_rad)


class BoardPoseEstimator:
    """Detect the plate tags and recover board tilt."""

    def __init__(
        self,
        calibration_path: str | Path,
        geometry: BoardGeometry,
        family: str = "tag36h11",
        min_tags: int = 3,
    ):
        calib = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        model = calib.get("kannala_brandt", calib)
        self.K = np.array(
            [
                [model["fx"], 0.0, model["cx"]],
                [0.0, model["fy"], model["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.asarray(model["dist"], dtype=np.float64)
        self.fisheye = "kannala_brandt" in calib or distortion.size == 4
        self.dist = distortion.reshape(4, 1) if self.fisheye else distortion.reshape(-1, 1)
        self.image_size = (int(calib["image_width"]), int(calib["image_height"]))
        self.calibration_path = str(calibration_path)

        # The live viewer uses the same intrinsic matrix as solvePnP, so an
        # undistorted pixel has exactly the same coordinates whether it came
        # from this image remap or from ``undistort(points)`` below.  Build the
        # maps once; rebuilding them for every video frame is needlessly slow.
        if self.fisheye:
            self._undistort_map_x, self._undistort_map_y = (
                cv2.fisheye.initUndistortRectifyMap(
                    self.K,
                    self.dist,
                    np.eye(3, dtype=np.float64),
                    self.K,
                    self.image_size,
                    cv2.CV_16SC2,
                )
            )
        else:
            self._undistort_map_x, self._undistort_map_y = (
                cv2.initUndistortRectifyMap(
                    self.K,
                    self.dist,
                    None,
                    self.K,
                    self.image_size,
                    cv2.CV_16SC2,
                )
            )

        self.geometry = geometry
        self.min_tags = int(min_tags)
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            APRILTAG_FAMILIES[family])
        self.params = self._detector_params(
            cv2.aruco.CORNER_REFINE_APRILTAG)
        self.roi_params = self._detector_params(
            cv2.aruco.CORNER_REFINE_CONTOUR)
        # APRILTAG refinement is excellent for native full-frame acquisition,
        # but can take seconds on a stale crop containing no marker. Once a
        # tag fills a small tracked ROI, CONTOUR is fast and precise; any miss
        # immediately falls back to the authoritative full-frame detector.
        # ArucoDetector copies the parameters at construction, so construct a
        # separate detector for each refinement mode.
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(
                self.dictionary, self.params)
            self.roi_detector = cv2.aruco.ArucoDetector(
                self.dictionary, self.roi_params)
        else:
            self.detector = None
            self.roi_detector = None
        self.zero_rotation: np.ndarray | None = None
        # Native-resolution tracking: a full-frame scan acquires/reacquires
        # the tags, then small per-tag crops preserve the same corner detail at
        # a fraction of the cost. Coordinates stored here are always in the
        # calibrated image space, independent of capture resolution.
        self.full_detection_interval = 15
        self._detection_calls = 0
        self._tracked_corners: dict[int, np.ndarray] = {}
        self.last_found: dict[int, np.ndarray] = {}
        self.last_detection_mode = "none"

    @staticmethod
    def _detector_params(refinement_method):
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()
        # Re-measured on the installed board after the corner tags were
        # rearranged (2026-08-10). APRILTAG recovered all four tags in 76% of
        # live frames versus 3% for CONTOUR, reduced median reprojection error
        # from 2.25 to 1.92 px, and gave about 0.007 deg static noise per axis.
        params.cornerRefinementMethod = refinement_method
        return params

    # ---- geometry helpers -------------------------------------------------
    def undistort(self, points: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,2) undistorted pixels in the same K."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(pts, self.K, self.dist, P=self.K)
        else:
            out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a full frame rectified into the solvePnP pixel space."""
        actual_size = (int(frame.shape[1]), int(frame.shape[0]))
        if actual_size != self.image_size:
            raise ValueError(
                f"frame is {actual_size[0]}x{actual_size[1]}, expected "
                f"calibrated size {self.image_size[0]}x{self.image_size[1]}"
            )
        return cv2.remap(
            frame,
            self._undistort_map_x,
            self._undistort_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def project_undistorted(
        self, points_board: np.ndarray, pose: PoseResult
    ) -> np.ndarray:
        """Project board-frame 3-D points into the rectified viewer image."""
        points = np.asarray(points_board, dtype=np.float64).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(pose.rotation)
        projected, _ = cv2.projectPoints(
            points,
            rvec,
            pose.translation.reshape(3, 1),
            self.K,
            None,
        )
        return projected.reshape(-1, 2)

    # ---- main -------------------------------------------------------------
    def _detect_full(
        self, gray: np.ndarray, pixel_scale: np.ndarray
    ) -> dict[int, np.ndarray]:
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.params)
        found: dict[int, np.ndarray] = {}
        if ids is None:
            return found
        for tag_id, corner in zip(ids.ravel(), corners):
            tag_id = int(tag_id)
            if tag_id in self.geometry.tags:
                found[tag_id] = corner[0].astype(np.float64) * pixel_scale
        return found

    def _detect_tracked_rois(
        self, gray: np.ndarray, pixel_scale: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Detect each expected tag near its previous native-image position."""
        inverse_scale = 1.0 / pixel_scale
        height, width = gray.shape[:2]
        found: dict[int, np.ndarray] = {}
        for expected_id in self.geometry.ids:
            calibrated = self._tracked_corners.get(expected_id)
            if calibrated is None:
                continue
            native = calibrated * inverse_scale
            low = native.min(axis=0)
            high = native.max(axis=0)
            tag_span = max(float(high[0] - low[0]), float(high[1] - low[1]))
            margin = max(24.0, 1.25 * tag_span)
            x0 = max(0, int(math.floor(low[0] - margin)))
            y0 = max(0, int(math.floor(low[1] - margin)))
            x1 = min(width, int(math.ceil(high[0] + margin)))
            y1 = min(height, int(math.ceil(high[1] + margin)))
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            crop = gray[y0:y1, x0:x1]
            if self.roi_detector is not None:
                corners, ids, _ = self.roi_detector.detectMarkers(crop)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    crop, self.dictionary, parameters=self.roi_params)
            if ids is None:
                continue
            for tag_id, corner in zip(ids.ravel(), corners):
                if int(tag_id) != expected_id:
                    continue
                native_corner = corner[0].astype(np.float64)
                native_corner += np.array([x0, y0], dtype=np.float64)
                found[expected_id] = native_corner * pixel_scale
                break
        return found

    def detect(self, gray: np.ndarray):
        """Return known tag corners in calibrated-image pixel coordinates.

        Detection may run on a larger image with the same aspect ratio. Corner
        refinement benefits substantially from the native 1920x1200 capture,
        while scaling the resulting pixels back to 1280x800 keeps PnP tied to
        the calibrated intrinsic matrix.
        """
        actual_height, actual_width = gray.shape[:2]
        expected_width, expected_height = self.image_size
        if actual_width * expected_height != actual_height * expected_width:
            raise ValueError(
                f"detection image {actual_width}x{actual_height} does not "
                f"match calibration aspect ratio {expected_width}x{expected_height}"
            )
        pixel_scale = np.array([
            expected_width / actual_width,
            expected_height / actual_height,
        ], dtype=np.float64)
        self._detection_calls += 1
        expected_ids = set(self.geometry.ids)
        periodic_full = (
            self._detection_calls % self.full_detection_interval == 0)
        can_track = set(self._tracked_corners) == expected_ids

        if can_track and not periodic_full:
            found = self._detect_tracked_rois(gray, pixel_scale)
            self.last_detection_mode = "roi"
            if set(found) != expected_ids:
                # The board may have moved outside a crop. Recover on this same
                # frame so tracking failure does not create a visible blink.
                found = self._detect_full(gray, pixel_scale)
                self.last_detection_mode = "recovery"
        else:
            found = self._detect_full(gray, pixel_scale)
            self.last_detection_mode = "full"

        if set(found) == expected_ids:
            self._tracked_corners = {
                tag_id: points.copy() for tag_id, points in found.items()
            }
        self.last_found = {
            tag_id: points.copy() for tag_id, points in found.items()
        }
        return found

    def estimate(self, gray: np.ndarray) -> PoseResult | None:
        found = self.detect(gray)
        if len(found) < self.min_tags:
            return None

        ids = tuple(sorted(found))
        image_points = np.concatenate([found[i] for i in ids], axis=0)
        object_points = self.geometry.object_points(ids)
        undistorted = self.undistort(image_points)

        ok, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float32),
            undistorted.astype(np.float32),
            self.K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        projected, _ = cv2.projectPoints(
            object_points.astype(np.float32), rvec, tvec, self.K, None)
        reprojection = float(
            np.sqrt(np.mean(np.sum(
                (projected.reshape(-1, 2) - undistorted) ** 2, axis=1))))

        rotation, _ = cv2.Rodrigues(rvec)
        relative = rotation if self.zero_rotation is None else (
            self.zero_rotation.T @ rotation)
        alpha, beta = angles_from_rotation(relative)

        return PoseResult(
            alpha_rad=alpha,
            beta_rad=beta,
            rotation=rotation,
            translation=tvec.reshape(3),
            tag_ids=ids,
            reprojection_px=reprojection,
            image_points=undistorted,
        )

    # ---- zeroing ----------------------------------------------------------
    def set_zero(self, rotation: np.ndarray) -> None:
        """Adopt this plate rotation as level. Valid until the camera moves."""
        self.zero_rotation = np.asarray(rotation, dtype=np.float64).copy()

    def clear_zero(self) -> None:
        self.zero_rotation = None

    def save_zero(self, path: str | Path) -> None:
        if self.zero_rotation is None:
            raise RuntimeError("no zero captured")
        Path(path).write_text(
            json.dumps({
                "zero_rotation": self.zero_rotation.tolist(),
                "calibration": self.calibration_path,
                "geometry": self.geometry.source,
                "note": (
                    "Plate rotation treated as level. Only valid while the "
                    "camera does not move; re-zero after any disturbance."
                ),
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_zero(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.zero_rotation = np.asarray(data["zero_rotation"], dtype=np.float64)
