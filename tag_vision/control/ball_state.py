"""Low-latency marble position/velocity filtering for real-hardware control."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallState:
    time_s: float
    position_mm: np.ndarray
    velocity_mm_s: np.ndarray
    measurement_age_s: float
    observed: bool

    @property
    def speed_mm_s(self) -> float:
        return float(np.linalg.norm(self.velocity_mm_s))


class BallStateFilter:
    """Alpha-beta filter with explicit dropout and reset handling.

    Camera positions are noisy enough that raw finite differences are unsuitable
    for feedback.  This filter predicts at every call and corrects only when a
    measurement is present. A long gap starts a fresh track instead of turning a
    reacquisition jump into a very large velocity estimate.
    """

    def __init__(self, *, position_gain: float = 0.35,
                 velocity_gain: float = 0.45,
                 reset_gap_s: float = 0.35,
                 measurement_noise_mm: float = 0.35,
                 stationary_speed_mm_s: float = 2.0,
                 velocity_window_s: float = 0.30,
                 stationary_displacement_mm: float = 3.0) -> None:
        if not 0 < position_gain <= 1 or not 0 < velocity_gain <= 1:
            raise ValueError("filter gains must be in (0, 1]")
        if reset_gap_s <= 0:
            raise ValueError("reset_gap_s must be positive")
        if (measurement_noise_mm < 0 or stationary_speed_mm_s < 0
                or velocity_window_s <= 0 or stationary_displacement_mm < 0):
            raise ValueError("noise and stationary thresholds cannot be negative")
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.reset_gap_s = float(reset_gap_s)
        self.measurement_noise_mm = float(measurement_noise_mm)
        self.stationary_speed_mm_s = float(stationary_speed_mm_s)
        self.velocity_window_s = float(velocity_window_s)
        self.stationary_displacement_mm = float(stationary_displacement_mm)
        self._position: np.ndarray | None = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._time: float | None = None
        self._measurement_time: float | None = None
        self._observations: deque[tuple[float, np.ndarray]] = deque()

    def reset(self) -> None:
        self._position = None
        self._velocity[:] = 0.0
        self._time = None
        self._measurement_time = None
        self._observations.clear()

    def _update_velocity(self, now: float, measurement: np.ndarray) -> None:
        self._observations.append((now, measurement.copy()))
        cutoff = now - self.velocity_window_s
        while len(self._observations) > 2 and self._observations[0][0] < cutoff:
            self._observations.popleft()
        if len(self._observations) < 2:
            self._velocity[:] = 0.0
            return
        rows = list(self._observations)
        split = max(1, len(rows) // 2)
        early = rows[:split]
        late = rows[split:]
        if not late:
            return
        early_position = np.median(np.stack([row[1] for row in early]), axis=0)
        late_position = np.median(np.stack([row[1] for row in late]), axis=0)
        early_time = float(np.median([row[0] for row in early]))
        late_time = float(np.median([row[0] for row in late]))
        span = late_time - early_time
        if span <= 1e-6:
            return
        displacement = late_position - early_position
        if np.linalg.norm(displacement) < self.stationary_displacement_mm:
            estimate = np.zeros(2, dtype=np.float64)
        else:
            estimate = displacement / span
        self._velocity = ((1.0 - self.velocity_gain) * self._velocity
                          + self.velocity_gain * estimate)
        if np.linalg.norm(self._velocity) < self.stationary_speed_mm_s:
            self._velocity[:] = 0.0

    def update(self, time_s: float, position_mm=None) -> BallState | None:
        now = float(time_s)
        measurement = None if position_mm is None else np.asarray(
            position_mm, dtype=np.float64)
        if measurement is not None and (
                measurement.shape != (2,) or not np.all(np.isfinite(measurement))):
            raise ValueError("position_mm must contain two finite values")

        if self._time is not None and now <= self._time:
            raise ValueError("timestamps must be strictly increasing")
        if self._position is None:
            self._time = now
            if measurement is None:
                return None
            self._position = measurement.copy()
            self._velocity[:] = 0.0
            self._measurement_time = now
            self._observations.append((now, measurement.copy()))
            return self._state(now, True)

        dt = now - float(self._time)
        predicted = self._position + self._velocity * dt
        observed = measurement is not None
        age_before = now - float(self._measurement_time)
        if measurement is not None and age_before > self.reset_gap_s:
            self._position = measurement.copy()
            self._velocity[:] = 0.0
            self._observations.clear()
            self._observations.append((now, measurement.copy()))
        elif measurement is not None:
            residual = measurement - predicted
            self._position = predicted + self.position_gain * residual
            self._update_velocity(now, measurement)
        else:
            self._position = predicted
        if measurement is not None:
            self._measurement_time = now
        self._time = now
        return self._state(now, observed)

    def _state(self, now: float, observed: bool) -> BallState:
        return BallState(
            time_s=now,
            position_mm=self._position.copy(),
            velocity_mm_s=self._velocity.copy(),
            measurement_age_s=now - float(self._measurement_time),
            observed=observed,
        )
