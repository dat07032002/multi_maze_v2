"""Deterministic reload brake for the repeatable physical entry trajectory."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .ball_state import BallState


class FixedResetPhase(str, Enum):
    WAITING_FOR_DROP = "waiting_for_drop"
    ARMED = "armed_for_reload"
    BRAKING = "fixed_braking"
    SETTLING = "reset_settling"
    READY = "episode_ready"
    TIMEOUT = "brake_timeout"


@dataclass(frozen=True)
class FixedResetCommand:
    phase: FixedResetPhase
    tilt_deg: np.ndarray
    episode_ready: bool


class FixedResetBrake:
    """Hold one calibrated backward tilt from ball loss through rest."""

    def __init__(self, brake_tilt_deg, *, trigger_speed_mm_s: float = 25.0,
                 settle_speed_mm_s: float = 12.0, settle_hold_s: float = 0.5,
                 max_brake_duration_s: float = 1.2,
                 minimum_brake_duration_s: float = 0.35,
                 arm_after_absent_s: float = 0.25,
                 stale_measurement_s: float = 0.12,
                 trigger_on_reappearance: bool = True) -> None:
        tilt = np.asarray(brake_tilt_deg, dtype=np.float64)
        if tilt.shape != (2,) or not np.all(np.isfinite(tilt)):
            raise ValueError("brake tilt must be a finite two-vector")
        if (trigger_speed_mm_s <= settle_speed_mm_s
                or settle_speed_mm_s < 0 or settle_hold_s <= 0
                or max_brake_duration_s <= minimum_brake_duration_s
                or minimum_brake_duration_s < 0 or arm_after_absent_s <= 0):
            raise ValueError("invalid fixed reset thresholds")
        self.brake_tilt_deg = tilt
        self.trigger_speed_mm_s = float(trigger_speed_mm_s)
        self.settle_speed_mm_s = float(settle_speed_mm_s)
        self.settle_hold_s = float(settle_hold_s)
        self.max_brake_duration_s = float(max_brake_duration_s)
        self.minimum_brake_duration_s = float(minimum_brake_duration_s)
        self.arm_after_absent_s = float(arm_after_absent_s)
        self.stale_measurement_s = float(stale_measurement_s)
        self.trigger_on_reappearance = bool(trigger_on_reappearance)
        self.reset()

    def reset(self) -> None:
        self._absent_since: float | None = None
        self._brake_since: float | None = None
        self._settle_since: float | None = None
        self._armed = False
        self._timed_out = False

    def update(self, state: BallState | None, now_s: float) -> FixedResetCommand:
        now = float(now_s)
        level = np.zeros(2, dtype=np.float64)
        stale = state is None or state.measurement_age_s > self.stale_measurement_s
        if stale:
            if self._absent_since is None:
                self._absent_since = now
            if now - self._absent_since >= self.arm_after_absent_s:
                if not self._armed or self._timed_out:
                    self._brake_since = None
                    self._settle_since = None
                    self._timed_out = False
                self._armed = True
            # Reload behavior is repeatable: once absence is confirmed, hold
            # the calibrated backward tilt while waiting for reappearance.
            # Before arming, remain level so an ordinary one-frame miss cannot
            # move the board.
            self._settle_since = None
            return FixedResetCommand(
                FixedResetPhase.ARMED if self._armed
                else FixedResetPhase.WAITING_FOR_DROP,
                self.brake_tilt_deg.copy() if self._armed else level, False)

        self._absent_since = None
        speed = state.speed_mm_s
        if self._timed_out:
            # Timeout levels the board but must not deadlock automatic
            # training. Once the marble is observed at rest, use the normal
            # settle hold and release a new episode. If it disappears again,
            # the stale branch above rearms a fresh brake attempt.
            if speed <= self.settle_speed_mm_s:
                if self._settle_since is None:
                    self._settle_since = now
                ready = now - self._settle_since >= self.settle_hold_s
                if ready:
                    self._armed = False
                    self._brake_since = None
                    self._settle_since = None
                    self._timed_out = False
                return FixedResetCommand(
                    FixedResetPhase.READY if ready
                    else FixedResetPhase.SETTLING, level, ready)
            self._settle_since = None
            return FixedResetCommand(FixedResetPhase.TIMEOUT, level, False)
        if not self._armed:
            return FixedResetCommand(FixedResetPhase.WAITING_FOR_DROP,
                                     level, False)

        if self._brake_since is None and (
                self.trigger_on_reappearance
                or speed >= self.trigger_speed_mm_s):
            self._brake_since = now
        if self._brake_since is not None:
            if now - self._brake_since > self.max_brake_duration_s:
                self._timed_out = True
                return FixedResetCommand(FixedResetPhase.TIMEOUT, level, False)
            if now - self._brake_since < self.minimum_brake_duration_s:
                return FixedResetCommand(FixedResetPhase.BRAKING,
                                         self.brake_tilt_deg.copy(), False)
            if speed > self.settle_speed_mm_s:
                self._settle_since = None
                return FixedResetCommand(FixedResetPhase.BRAKING,
                                         self.brake_tilt_deg.copy(), False)

        if speed <= self.settle_speed_mm_s:
            if self._settle_since is None:
                self._settle_since = now
            ready = now - self._settle_since >= self.settle_hold_s
            if ready:
                self._armed = False
                self._brake_since = None
                self._settle_since = None
            return FixedResetCommand(
                FixedResetPhase.READY if ready else FixedResetPhase.SETTLING,
                level, ready)
        return FixedResetCommand(FixedResetPhase.ARMED, level, False)
