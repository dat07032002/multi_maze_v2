"""A hand-written controller that drives the maze, for three jobs.

**A feasibility check on the hardware before any RL.** If a controller with a
correct model, a validated actuator and perfect state cannot get round, the
problem is the rig -- most likely that level sits 2.4 degrees off the midpoint of
roll's travel -- and no amount of training will fix it. Finding that out costs a
day here and weeks later.

**A bar for RL to clear.** "80 % success" means nothing on its own. "Beats the
analytic controller" is a claim about whether learning bought anything.

**A base policy for the residual stage.** M7 freezes something and learns a
bounded correction on top. A controller built from measured physics has no
sim-to-real gap in its *structure*, only in its parameters.

The control law is deliberately plain. The plant is a double integrator once the
predictor has removed the delay, so pure pursuit toward a lookahead point with
velocity damping is close to the best a fixed law can do; the interesting part
is not the feedback but the speed reference, which has to respect three separate
limits at once -- cornering, clearance, and stopping at the goal.
"""
from __future__ import annotations

import math

import numpy as np

from sim.analytic_model import ACCEL_PER_RAD


class PurePursuitBaseline:
    """Follow an arc-length route on a double integrator."""

    def __init__(self, route, max_tilt_rad: float,
                 speed_max: float = 0.040,
                 velocity_gain: float = 9.0,
                 cross_track_gain: float = 3.0,
                 cross_track_clamp_m: float = 0.006,
                 clearance_reference_m: float = 0.004,
                 goal_radius_m: float = 0.008):
        # Tuned over a 10-seed sweep. The working region is narrow and the
        # boundaries are sharp: 45 mm/s drops to 8/10, cross-track gain 4.0 to
        # 7/10, velocity gain 12 to 7/10. That narrowness is itself the result
        # -- a hand-tuned law on a plant with 240 ms of loop delay has very
        # little margin, which is most of the argument for learning one.
        self.route = route
        self.max_tilt = max_tilt_rad
        self.speed_max = speed_max
        self.kv = velocity_gain
        self.kx = cross_track_gain
        self.cross_clamp = cross_track_clamp_m
        self.clearance_reference = clearance_reference_m
        self.goal_radius = goal_radius_m
        self.accel_max = ACCEL_PER_RAD * math.sin(max_tilt_rad)

    # ---- speed reference ----------------------------------------------
    def _speed_reference(self, s: float) -> float:
        """Slowest of the three limits that apply at this point on the route.

        *Stopping*: enough room left to brake to rest at the goal.
        *Clearance*: the route passes within 1.95 mm of a wall at its tightest
        and has 30 mm in open cells; the same speed cannot be right for both.
        *Cornering*: lateral acceleration is drawn from the same budget as
        forward acceleration, so a turn has to be entered slowly enough that
        holding the line does not need more tilt than exists.
        """
        remaining = max(0.0, self.route.length - s)
        stopping = math.sqrt(2.0 * self.accel_max * remaining)

        clearance = max(0.0, self.route.clearance_at(s))
        room = min(1.0, clearance / self.clearance_reference)

        radius = self._turn_radius(s)
        cornering = math.sqrt(self.accel_max * radius) if radius > 0 else self.speed_max

        return min(self.speed_max * room, stopping, cornering)

    def _turn_radius(self, s: float) -> float:
        """Local radius of curvature, from the tangent turning over 20 mm."""
        route = self.route
        span = max(1, int(0.020 / route.spacing))
        i = route.index_at(s)
        a = route.tangents[max(0, i - span)]
        b = route.tangents[min(len(route.tangents) - 1, i + span)]
        turn = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
        if turn < 1e-6:
            return math.inf
        return (2 * span * route.spacing) / turn

    # ---- control law ---------------------------------------------------
    def __call__(self, position, velocity) -> tuple[float, float]:
        position = np.asarray(position, dtype=float)[:2]
        velocity = np.asarray(velocity, dtype=float)[:2]

        s, cross, index = self.route.project(position)
        if np.linalg.norm(position - self.route.goal) < self.goal_radius:
            # Arrived: kill the remaining speed rather than driving through.
            return self._to_tilt(-self.kv * velocity)

        # A velocity field along the route, rather than pure pursuit toward a
        # distant point. In 23 mm cells a lookahead long enough to be smooth is
        # longer than the cell, so it points through walls on every corner and
        # the ball is steered into them. Using the tangent at the projection
        # keeps the reference on the path by construction.
        tangent = self.route.tangents[index]
        normal = self.route.normals[index]
        correction = float(np.clip(cross, -self.cross_clamp, self.cross_clamp))
        desired = (tangent * self._speed_reference(s)
                   - normal * self.kx * correction)

        accel = self.kv * (desired - velocity)
        return self._to_tilt(accel)

    def _to_tilt(self, accel) -> tuple[float, float]:
        """Invert ``x'' = 7.007 sin(beta)``, ``y'' = -7.007 sin(alpha)``."""
        scaled = np.clip(np.asarray(accel) / ACCEL_PER_RAD, -1.0, 1.0)
        beta = math.asin(float(scaled[0]))
        alpha = math.asin(float(-scaled[1]))
        return (float(np.clip(alpha, -self.max_tilt, self.max_tilt)),
                float(np.clip(beta, -self.max_tilt, self.max_tilt)))
