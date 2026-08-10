"""Slow bounded fused-angle feedback for settled static positioning."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .directional_calibration import DirectionalMotorCalibration


@dataclass(frozen=True)
class TrimUpdate:
    counts: dict[int, int]
    delta_counts: tuple[int, int]
    error_deg: tuple[float, float]
    converged: bool = False
    exhausted: bool = False


class FusedStaticTrim:
    """Iteratively remove settled fused error without becoming a fast PID.

    The feedforward move does the real work. This helper waits for the plate to
    settle, applies at most a few bounded count corrections through the inverse
    measured 2x2 Jacobian, and then stops. It therefore cannot integrate while
    the linkage is moving or wind up indefinitely inside backlash.
    """

    def __init__(
        self,
        calibration: DirectionalMotorCalibration,
        *,
        tolerance_deg: float = 0.10,
        settle_delay_s: float = 0.8,
        convergence_hold_s: float = 0.4,
        gain: float = 0.7,
        max_step_counts: int = 80,
        min_step_counts: int = 20,
        max_iterations: int = 6,
    ) -> None:
        if tolerance_deg <= 0 or settle_delay_s < 0 or convergence_hold_s < 0:
            raise ValueError("invalid trim tolerance or delay")
        if not 0 < gain <= 1 or max_step_counts <= 0 or max_iterations <= 0:
            raise ValueError("invalid trim gain, step, or iteration limit")
        self.calibration = calibration
        self.tolerance_deg = float(tolerance_deg)
        self.settle_delay_s = float(settle_delay_s)
        self.convergence_hold_s = float(convergence_hold_s)
        self.gain = float(gain)
        self.max_step_counts = int(max_step_counts)
        self.min_step_counts = int(min_step_counts)
        self.max_iterations = int(max_iterations)
        matrix = np.asarray(calibration.jacobian_deg_per_count,
                            dtype=np.float64)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("trim Jacobian must be a finite 2x2 matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-8:
            raise ValueError("trim Jacobian is singular or poorly scaled")
        self._inverse = np.linalg.inv(matrix)
        self.active = False
        self.target = np.zeros(2)
        self.goal_counts: dict[int, int] = {}
        self.last_move_time = 0.0
        self.iterations = 0
        self._within_since: float | None = None

    def arm(self, target_deg, goal_counts: dict[int, int], timestamp: float) -> None:
        target = np.asarray(target_deg, dtype=np.float64)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ValueError("trim target must contain two finite angles")
        self.target = target
        self.goal_counts = dict(goal_counts)
        self.last_move_time = float(timestamp)
        self.iterations = 0
        self._within_since = None
        self.active = True

    def update(self, measured_deg, *, timestamp: float,
               settled: bool) -> TrimUpdate | None:
        if not self.active or not settled:
            return None
        now = float(timestamp)
        if now - self.last_move_time < self.settle_delay_s:
            return None
        measured = np.asarray(measured_deg, dtype=np.float64)
        if measured.shape != (2,) or not np.all(np.isfinite(measured)):
            return None
        error = self.target - measured
        error_tuple = (float(error[0]), float(error[1]))
        if float(np.max(np.abs(error))) <= self.tolerance_deg:
            if self._within_since is None:
                self._within_since = now
            if now - self._within_since >= self.convergence_hold_s:
                self.active = False
                return TrimUpdate(dict(self.goal_counts), (0, 0), error_tuple,
                                  converged=True)
            return None
        self._within_since = None
        if self.iterations >= self.max_iterations:
            self.active = False
            return TrimUpdate(dict(self.goal_counts), (0, 0), error_tuple,
                              exhausted=True)

        raw = self.gain * (self._inverse @ error)
        raw = np.clip(raw, -self.max_step_counts, self.max_step_counts)
        for index in range(2):
            if abs(error[index]) <= self.tolerance_deg:
                # Cross terms on this rig are tiny. Do not disturb an axis
                # already on target merely to improve the other by <0.01 deg.
                raw[index] = 0.0
            elif 1e-9 < abs(raw[index]) < self.min_step_counts:
                raw[index] = np.sign(raw[index]) * self.min_step_counts
        requested = np.rint(raw).astype(int)
        axes = (self.calibration.alpha, self.calibration.beta)
        updated = dict(self.goal_counts)
        actual = []
        for axis, change in zip(axes, requested):
            old = updated[axis.servo_id]
            new = int(min(max(old + int(change), axis.min_counts), axis.max_counts))
            updated[axis.servo_id] = new
            actual.append(new - old)
        self.goal_counts = updated
        self.iterations += 1
        self.last_move_time = now
        self._within_since = None
        return TrimUpdate(dict(updated), (actual[0], actual[1]), error_tuple)
