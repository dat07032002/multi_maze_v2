"""Propagate the ball across the dead time, so the policy sees *now*.

The single most valuable thing the analytic model buys. At the +-4 degree
command limit the ball crosses one 23 mm cell in 0.31 s, while the actuator
needs 0.19 s of dead time plus 0.26 s of rise before the commanded tilt is
actually on the board -- and the camera adds an unmeasured 30-120 ms on top. A
controller acting on the measured state is acting on where the ball was a third
of a cell ago.

Nothing about that is unknowable, though. The commands already sent are known
exactly, the actuator's response to them is modelled and validated to 0.5 %, and
the ball's motion between contacts is closed-form. So the state at the moment
the next command will actually land can be *computed* rather than learned. That
converts a delayed MDP into a nearly undelayed one, which is why the observation
carries 3 past actions instead of 10.

Two honest limits. The propagation uses the between-contact model, so a
prediction that crosses a wall impact is wrong -- bounded by how far the ball
travels in the delay, up to about 150 mm at full speed, though in a 20 mm
corridor it is usually much less. And it inherits the camera latency, which is
still assumed rather than measured.
"""
from __future__ import annotations

import copy

import numpy as np

from sim import analytic_model


class StatePredictor:
    """Rolls the ball forward through the actuator's already-committed motion."""

    def __init__(self, actuator, timestep: float, sensor_latency_s: float = 0.0,
                 damping: float = analytic_model.DEFAULT_DAMPING,
                 coulomb: float = analytic_model.DEFAULT_COULOMB):
        self.actuator = actuator
        self.dt = timestep
        self.sensor_latency_s = float(sensor_latency_s)
        self.damping = damping
        self.coulomb = coulomb

    def horizon_steps(self) -> int:
        """How far ahead to predict: sensor latency plus actuator dead time.

        Both matter and they add. The measurement describes the past by the
        camera's latency; the command will not take effect for the actuator's
        dead time. The gap between what you know and what you can affect is the
        sum.
        """
        dead = max(plant.dyn.dead_time_s for plant in self.actuator.plants.values())
        return int(round((self.sensor_latency_s + dead) / self.dt))

    def future_angles(self, held_command) -> list[tuple[float, float]]:
        """Board angles over the horizon if the current command were held.

        Uses a *copy* of the actuator so asking what will happen does not
        advance what is happening -- a mistake that would silently double the
        commanded motion every control step.
        """
        model = copy.deepcopy(self.actuator)
        return [model.step(*held_command) for _ in range(self.horizon_steps())]

    def predict(self, position, velocity, held_command):
        """Ball state at the moment the next command can first take effect."""
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        for alpha, beta in self.future_angles(held_command):
            position, velocity = analytic_model.step(
                position, velocity, alpha, beta, self.dt, self.damping,
                self.coulomb)
        return position, velocity
