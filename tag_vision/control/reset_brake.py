"""Adaptive reset braking before a new real-hardware episode starts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .ball_dynamics import BallDynamicsModel
from .ball_state import BallState


class ResetPhase(str, Enum):
    WAITING = "waiting_for_ball"
    BRAKING = "braking"
    SETTLING = "settling"
    READY = "ready"
    LOST = "ball_lost"
    TIMEOUT = "brake_timeout"


@dataclass(frozen=True)
class ResetCommand:
    phase: ResetPhase
    tilt_deg: np.ndarray
    episode_ready: bool
    reason: str


class AdaptiveResetBrake:
    """Oppose measured velocity, level, then require a stable hold.

    The action is clipped in acceleration and angle. Missing/stale vision always
    returns level; it never continues a blind braking command.
    """

    def __init__(self, model: BallDynamicsModel, *,
                 velocity_time_constant_s: float = 0.22,
                 max_brake_accel_mm_s2: float = 900.0,
                 max_tilt_deg: float = 1.25,
                 enter_speed_mm_s: float = 25.0,
                 settle_speed_mm_s: float = 12.0,
                 settle_hold_s: float = 0.50,
                 max_measurement_age_s: float = 0.12,
                 max_brake_duration_s: float = 1.5) -> None:
        if velocity_time_constant_s <= 0 or max_brake_accel_mm_s2 <= 0:
            raise ValueError("brake constants must be positive")
        if (max_tilt_deg <= 0 or settle_speed_mm_s < 0 or settle_hold_s <= 0
                or max_measurement_age_s <= 0 or max_brake_duration_s <= 0):
            raise ValueError("invalid reset limits")
        self.model = model
        self.velocity_time_constant_s = float(velocity_time_constant_s)
        self.max_brake_accel_mm_s2 = float(max_brake_accel_mm_s2)
        self.max_tilt_deg = float(max_tilt_deg)
        self.enter_speed_mm_s = float(enter_speed_mm_s)
        self.settle_speed_mm_s = float(settle_speed_mm_s)
        self.settle_hold_s = float(settle_hold_s)
        self.max_measurement_age_s = float(max_measurement_age_s)
        self.max_brake_duration_s = float(max_brake_duration_s)
        self._settle_since: float | None = None
        self._brake_started: float | None = None
        self._seen = False

    def reset(self) -> None:
        self._settle_since = None
        self._brake_started = None
        self._seen = False

    def update(self, state: BallState | None) -> ResetCommand:
        level = np.zeros(2, dtype=np.float64)
        if state is None:
            return ResetCommand(ResetPhase.WAITING, level, False,
                                "waiting for first ball detection")
        if state.measurement_age_s > self.max_measurement_age_s:
            self._settle_since = None
            self._brake_started = None
            phase = ResetPhase.LOST if self._seen else ResetPhase.WAITING
            return ResetCommand(phase, level, False,
                                "vision stale; commanding level")
        self._seen = True
        speed = state.speed_mm_s
        if speed <= self.settle_speed_mm_s:
            self._brake_started = None
            if self._settle_since is None:
                self._settle_since = state.time_s
            ready = state.time_s - self._settle_since >= self.settle_hold_s
            return ResetCommand(
                ResetPhase.READY if ready else ResetPhase.SETTLING,
                level, ready,
                "ball stable" if ready else "holding level until stable")
        self._settle_since = None
        # Do not chatter into active braking on a small velocity fluctuation.
        if speed < self.enter_speed_mm_s:
            self._brake_started = None
            return ResetCommand(ResetPhase.SETTLING, level, False,
                                "below brake-entry speed")
        if self._brake_started is None:
            self._brake_started = state.time_s
        elif state.time_s - self._brake_started > self.max_brake_duration_s:
            return ResetCommand(ResetPhase.TIMEOUT, level, False,
                                "brake timed out; commanding level")
        acceleration = -state.velocity_mm_s / self.velocity_time_constant_s
        magnitude = float(np.linalg.norm(acceleration))
        if magnitude > self.max_brake_accel_mm_s2:
            acceleration *= self.max_brake_accel_mm_s2 / magnitude
        tilt = self.model.tilt_for_acceleration(
            acceleration, state.velocity_mm_s)
        tilt = np.clip(tilt, -self.max_tilt_deg, self.max_tilt_deg)
        return ResetCommand(ResetPhase.BRAKING, tilt, False,
                            "opposing measured entry velocity")
