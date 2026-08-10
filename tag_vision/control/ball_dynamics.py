"""Small identified model of marble motion on the physical maze."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BallDynamicsModel:
    """Continuous model ``accel = B @ tilt - D @ velocity + bias``.

    Position is millimetres, time seconds, and tilt degrees. Keeping the fitted
    units human-readable also makes an obviously bad identification easy to
    spot before it is allowed to command hardware.
    """

    acceleration_per_tilt: np.ndarray
    velocity_damping: np.ndarray
    bias_mm_s2: np.ndarray
    fit_rmse_mm_s2: float = float("nan")

    def __post_init__(self) -> None:
        for name, value, shape in (
            ("acceleration_per_tilt", self.acceleration_per_tilt, (2, 2)),
            ("velocity_damping", self.velocity_damping, (2, 2)),
            ("bias_mm_s2", self.bias_mm_s2, (2,)),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must have shape {shape} and be finite")
            object.__setattr__(self, name, array)
        if abs(float(np.linalg.det(self.acceleration_per_tilt))) < 1e-6:
            raise ValueError("acceleration_per_tilt is singular")

    def acceleration(self, velocity_mm_s, tilt_deg) -> np.ndarray:
        velocity = np.asarray(velocity_mm_s, dtype=np.float64)
        tilt = np.asarray(tilt_deg, dtype=np.float64)
        return (self.acceleration_per_tilt @ tilt
                - self.velocity_damping @ velocity + self.bias_mm_s2)

    def step(self, state, tilt_deg, dt_s: float) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (4,) or dt_s <= 0:
            raise ValueError("state must be [x,y,vx,vy] and dt_s positive")
        acceleration = self.acceleration(state[2:], tilt_deg)
        result = state.copy()
        result[:2] += state[2:] * dt_s + 0.5 * acceleration * dt_s ** 2
        result[2:] += acceleration * dt_s
        return result

    def tilt_for_acceleration(self, acceleration_mm_s2,
                              velocity_mm_s=(0.0, 0.0)) -> np.ndarray:
        desired = np.asarray(acceleration_mm_s2, dtype=np.float64)
        velocity = np.asarray(velocity_mm_s, dtype=np.float64)
        return np.linalg.solve(
            self.acceleration_per_tilt,
            desired + self.velocity_damping @ velocity - self.bias_mm_s2)

    def save(self, path: str | Path) -> None:
        data = {
            "version": "tag_ball_dynamics_v1",
            "units": {"position": "mm", "time": "s", "tilt": "deg"},
            "acceleration_per_tilt": self.acceleration_per_tilt.tolist(),
            "velocity_damping": self.velocity_damping.tolist(),
            "bias_mm_s2": self.bias_mm_s2.tolist(),
            "fit_rmse_mm_s2": self.fit_rmse_mm_s2,
        }
        Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BallDynamicsModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != "tag_ball_dynamics_v1":
            raise ValueError("unsupported ball dynamics model")
        return cls(np.asarray(data["acceleration_per_tilt"]),
                   np.asarray(data["velocity_damping"]),
                   np.asarray(data["bias_mm_s2"]),
                   float(data.get("fit_rmse_mm_s2", float("nan"))))


def fit_ball_dynamics(velocity_mm_s, tilt_deg, acceleration_mm_s2,
                      *, ridge: float = 1e-6) -> BallDynamicsModel:
    """Fit both axes jointly; cross-axis coupling is deliberately retained."""
    velocity = np.asarray(velocity_mm_s, dtype=np.float64)
    tilt = np.asarray(tilt_deg, dtype=np.float64)
    acceleration = np.asarray(acceleration_mm_s2, dtype=np.float64)
    if (velocity.ndim != 2 or velocity.shape[1] != 2
            or tilt.shape != velocity.shape or acceleration.shape != velocity.shape
            or len(velocity) < 8):
        raise ValueError("need at least 8 aligned Nx2 velocity/tilt/acceleration rows")
    design = np.column_stack((tilt, -velocity, np.ones(len(velocity))))
    regularizer = ridge * np.eye(design.shape[1])
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer, design.T @ acceleration)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean((predicted - acceleration) ** 2)))
    return BallDynamicsModel(
        acceleration_per_tilt=coefficients[:2].T,
        velocity_damping=coefficients[2:4].T,
        bias_mm_s2=coefficients[4], fit_rmse_mm_s2=rmse)
