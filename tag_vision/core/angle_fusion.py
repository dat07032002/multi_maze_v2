"""Complementary fusion of camera and IMU board-angle estimates.

The IMU contributes incremental motion, so the fused estimate remains smooth
and responsive between video frames or while AprilTags are briefly missing.
The camera contributes the absolute reference, slowly removing IMU drift.  A
camera outlier gate prevents a bad pose from dragging the estimate away.

Angles are in degrees because both live tools display and log degrees.  This is
an estimator boundary; control code can convert its output to radians once.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _angles(value) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        return None
    return result


def _wrapped_delta_deg(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference, component by component."""
    return (current - previous + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class FusedAngles:
    alpha_deg: float
    beta_deg: float
    camera_used: tuple[bool, bool]
    imu_used: bool
    camera_residual_deg: tuple[float, float]

    @property
    def array(self) -> np.ndarray:
        return np.array([self.alpha_deg, self.beta_deg], dtype=np.float64)


class CameraImuFusion:
    """Two-axis complementary filter with camera dropout/outlier handling.

    ``camera_time_constant_s`` controls how quickly the absolute camera pose
    corrects the IMU-propagated state. At the default 0.5 s, about 86% of a
    persistent offset is removed in one second. ``imu_timestamp`` is required
    to distinguish a new IMU report from the same report observed by several
    camera frames; an increment is never integrated twice.
    """

    def __init__(
        self,
        camera_time_constant_s: float = 0.5,
        max_camera_residual_deg: float = 2.0,
    ) -> None:
        if camera_time_constant_s <= 0:
            raise ValueError("camera_time_constant_s must be positive")
        if max_camera_residual_deg <= 0:
            raise ValueError("max_camera_residual_deg must be positive")
        self.camera_time_constant_s = float(camera_time_constant_s)
        self.max_camera_residual_deg = float(max_camera_residual_deg)
        self._state: np.ndarray | None = None
        self._last_imu: np.ndarray | None = None
        self._last_imu_timestamp: float | int | None = None
        self._last_update_time: float | None = None

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def angles_deg(self) -> np.ndarray | None:
        return None if self._state is None else self._state.copy()

    def reset(
        self,
        camera_angles_deg=None,
        imu_angles_deg=None,
        *,
        timestamp: float | None = None,
        imu_timestamp: float | int | None = None,
    ) -> None:
        camera = _angles(camera_angles_deg)
        imu = _angles(imu_angles_deg)
        self._state = (camera.copy() if camera is not None else
                       imu.copy() if imu is not None else None)
        self._last_imu = None if imu is None else imu.copy()
        self._last_imu_timestamp = imu_timestamp if imu is not None else None
        self._last_update_time = timestamp

    def update(
        self,
        camera_angles_deg=None,
        imu_angles_deg=None,
        *,
        timestamp: float,
        imu_timestamp: float | int | None = None,
    ) -> FusedAngles | None:
        camera = _angles(camera_angles_deg)
        imu = _angles(imu_angles_deg)
        now = float(timestamp)
        imu_used = False

        if self._state is None:
            self.reset(camera, imu, timestamp=now,
                       imu_timestamp=imu_timestamp)
            if self._state is None:
                return None
            return FusedAngles(
                float(self._state[0]), float(self._state[1]),
                (camera is not None, camera is not None), False, (0.0, 0.0))

        new_imu = (imu is not None and (
            imu_timestamp is None
            or self._last_imu_timestamp is None
            or imu_timestamp != self._last_imu_timestamp))
        if new_imu and self._last_imu is not None:
            self._state += _wrapped_delta_deg(imu, self._last_imu)
            imu_used = True
        if new_imu:
            self._last_imu = imu.copy()
            self._last_imu_timestamp = imu_timestamp

        residual = np.zeros(2, dtype=np.float64)
        camera_used = np.zeros(2, dtype=bool)
        if camera is not None:
            residual = _wrapped_delta_deg(camera, self._state)
            camera_used = np.abs(residual) <= self.max_camera_residual_deg
            if self._last_update_time is None:
                gain = 1.0
            else:
                dt = max(0.0, now - self._last_update_time)
                gain = 1.0 - math.exp(-dt / self.camera_time_constant_s)
            self._state[camera_used] += gain * residual[camera_used]

        self._last_update_time = now
        return FusedAngles(
            float(self._state[0]), float(self._state[1]),
            (bool(camera_used[0]), bool(camera_used[1])), imu_used,
            (float(residual[0]), float(residual[1])))
