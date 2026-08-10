"""Independent health classification and action safety shield."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class HealthLevel(IntEnum):
    GREEN = 0
    YELLOW = 1
    RED = 2


@dataclass(frozen=True)
class HealthSnapshot:
    level: HealthLevel
    reasons: tuple[str, ...]
    camera_fps: float
    pose_rate: float
    ball_rate: float
    ball_age_s: float
    imu_rate_hz: float
    fusion_residual_deg: float
    control_latency_s: float
    servo_timeout_count: int = 0
    servo_load_max: float = 0.0


@dataclass(frozen=True)
class SafeAction:
    proposed_deg: np.ndarray
    executed_deg: np.ndarray
    overridden: bool
    reason: str | None


class HealthMonitor:
    def __init__(self, *, max_tilt_deg: float = 4.0,
                 max_delta_deg: float = 0.5,
                 min_camera_fps: float = 15.0,
                 min_pose_rate: float = 0.80,
                 min_ball_rate: float = 0.85,
                 max_ball_age_s: float = 0.25,
                 min_imu_rate_hz: float = 180.0,
                 max_fusion_residual_deg: float = 2.0,
                 max_control_latency_s: float = 0.20,
                 max_servo_load: float = 350.0) -> None:
        self.max_tilt_deg = float(max_tilt_deg)
        self.max_delta_deg = float(max_delta_deg)
        self.min_camera_fps = float(min_camera_fps)
        self.min_pose_rate = float(min_pose_rate)
        self.min_ball_rate = float(min_ball_rate)
        self.max_ball_age_s = float(max_ball_age_s)
        self.min_imu_rate_hz = float(min_imu_rate_hz)
        self.max_fusion_residual_deg = float(max_fusion_residual_deg)
        self.max_control_latency_s = float(max_control_latency_s)
        self.max_servo_load = float(max_servo_load)
        self.previous_action = np.zeros(2, dtype=np.float64)

    def classify(self, **metrics) -> HealthSnapshot:
        reasons: list[str] = []
        level = HealthLevel.GREEN

        def flag(condition: bool, reason: str, severity: HealthLevel) -> None:
            nonlocal level
            if condition:
                reasons.append(reason)
                level = max(level, severity)

        camera_fps = float(metrics.get("camera_fps", 0.0))
        pose_rate = float(metrics.get("pose_rate", 0.0))
        ball_rate = float(metrics.get("ball_rate", 0.0))
        ball_age = float(metrics.get("ball_age_s", math.inf))
        imu_rate = float(metrics.get("imu_rate_hz", 0.0))
        residual = float(metrics.get("fusion_residual_deg", math.inf))
        latency = float(metrics.get("control_latency_s", math.inf))
        timeouts = int(metrics.get("servo_timeout_count", 0))
        load = float(metrics.get("servo_load_max", 0.0))
        flag(camera_fps < self.min_camera_fps, "camera_fps", HealthLevel.YELLOW)
        flag(pose_rate < self.min_pose_rate, "pose_rate", HealthLevel.YELLOW)
        flag(ball_rate < self.min_ball_rate, "ball_rate", HealthLevel.YELLOW)
        flag(ball_age > self.max_ball_age_s, "ball_stale", HealthLevel.RED)
        flag(imu_rate < self.min_imu_rate_hz, "imu_rate", HealthLevel.YELLOW)
        flag(residual > self.max_fusion_residual_deg,
             "fusion_residual", HealthLevel.RED)
        flag(latency > self.max_control_latency_s,
             "control_deadline", HealthLevel.RED)
        flag(timeouts > 0, "servo_timeout", HealthLevel.RED)
        flag(load > self.max_servo_load, "servo_load", HealthLevel.RED)
        return HealthSnapshot(level, tuple(reasons), camera_fps, pose_rate,
                              ball_rate, ball_age, imu_rate, residual, latency,
                              timeouts, load)

    def safe_action(self, proposed_deg, health: HealthSnapshot) -> SafeAction:
        proposed = np.asarray(proposed_deg, dtype=np.float64)
        if proposed.shape != (2,) or not np.all(np.isfinite(proposed)):
            executed = np.zeros(2, dtype=np.float64)
            self.previous_action = executed
            return SafeAction(proposed, executed, True, "invalid_action")
        if health.level == HealthLevel.RED:
            executed = np.zeros(2, dtype=np.float64)
            self.previous_action = executed
            return SafeAction(proposed.copy(), executed, True,
                              ",".join(health.reasons) or "red_health")
        clipped = np.clip(proposed, -self.max_tilt_deg, self.max_tilt_deg)
        delta = np.clip(clipped - self.previous_action,
                        -self.max_delta_deg, self.max_delta_deg)
        executed = self.previous_action + delta
        overridden = not np.allclose(executed, proposed)
        reason = "tilt_or_rate_limit" if overridden else None
        self.previous_action = executed.copy()
        return SafeAction(proposed.copy(), executed, overridden, reason)

    def reset_action(self) -> None:
        self.previous_action[:] = 0.0
