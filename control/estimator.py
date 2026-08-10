"""Kalman filter for the ball, one decoupled axis at a time.

Needed on hardware regardless of what the policy is. The detector reports
position only, so velocity has to come from somewhere, and differencing it is
not good enough: at 1 mm of position noise and a 50 ms control period, a finite
difference carries **28 mm/s** of noise against ball speeds of 0-500 mm/s, and
56 mm/s if the detector turns out to be 2 mm. A policy fed that is being asked
to control on a signal that is a tenth noise.

The filter earns its keep twice over by also riding through dropouts. The
detector loses the ball sometimes -- occlusion, a blue distractor, the ball
against a rail -- and a predicted position through a missing frame is far better
than a held stale one, because the model knows the ball is still accelerating
down the tilt.

Axes are independent: ``x`` responds to ``beta`` and ``y`` to ``alpha``, with no
cross terms, because the plate-rotation coupling is 0.54 % (see
``sim.analytic_model``). Two 2-state filters, not one 4-state, so the structure
is visible rather than buried in a block-diagonal matrix.
"""
from __future__ import annotations

import numpy as np

from sim import analytic_model


class AxisKalman:
    """Constant-acceleration filter on ``[position, velocity]``."""

    def __init__(self, process_accel_std: float, measurement_std: float):
        self.q = float(process_accel_std)
        self.r = float(measurement_std) ** 2
        self.x = np.zeros(2)
        self.P = np.diag([1e-2, 1e-1])

    def reset(self, position: float = 0.0, velocity: float = 0.0) -> None:
        self.x = np.array([position, velocity], dtype=float)
        self.P = np.diag([1e-2, 1e-1])

    def predict(self, accel: float, dt: float, damping: float = 0.0) -> None:
        # Damping belongs in F, not in the control input: it acts on the state.
        F = np.array([[1.0, dt], [0.0, 1.0 - damping * dt]])
        self.x = F @ self.x + np.array([0.5 * dt * dt, dt]) * accel
        # Process noise from an unmodelled acceleration acting over dt: rolling
        # resistance, wall contact, the gain error, everything the double
        # integrator does not know about.
        G = np.array([[0.5 * dt * dt], [dt]])
        self.P = F @ self.P @ F.T + G @ G.T * (self.q ** 2)

    def update(self, measurement: float) -> None:
        # Written out in scalars rather than with a (1,2) H: position is the
        # only observation, so the matrix form is all bookkeeping around
        # P[0, 0] and it obscures what is happening.
        innovation = float(measurement) - self.x[0]
        S = self.P[0, 0] + self.r
        K = self.P[:, 0] / S
        self.x = self.x + K * innovation
        self.P = self.P - np.outer(K, self.P[0, :])

    @property
    def position(self) -> float:
        return float(self.x[0])

    @property
    def velocity(self) -> float:
        return float(self.x[1])


class BallEstimator:
    """Both axes, driven by the board angle and corrected by the detector."""

    def __init__(self, process_accel_std: float = 0.08,
                 measurement_std: float = 0.001,
                 damping: float = analytic_model.DEFAULT_DAMPING):
        # 0.08 m/s^2 of unmodelled acceleration, against a 0.489 m/s^2 budget at
        # full tilt. The model is good to 0.138 mm over the prediction horizon,
        # so most of this allowance is for wall contacts, which the double
        # integrator knows nothing about. An earlier 0.35 made the filter trust
        # the detector almost completely and handed the controller a velocity
        # estimate barely better than a finite difference.
        self.x = AxisKalman(process_accel_std, measurement_std)
        self.y = AxisKalman(process_accel_std, measurement_std)
        self.damping = damping

    def reset(self, position=(0.0, 0.0), velocity=(0.0, 0.0)) -> None:
        self.x.reset(position[0], velocity[0])
        self.y.reset(position[1], velocity[1])

    def predict(self, alpha: float, beta: float, dt: float) -> None:
        ax, ay = analytic_model.acceleration(alpha, beta)
        self.x.predict(ax, dt, self.damping)
        self.y.predict(ay, dt, self.damping)

    def update(self, measurement) -> None:
        """Fold in a detector reading. Pass ``None`` for a dropped frame."""
        if measurement is None:
            return
        self.x.update(float(measurement[0]))
        self.y.update(float(measurement[1]))

    @property
    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.x.position, self.y.position]),
                np.array([self.x.velocity, self.y.velocity]))
