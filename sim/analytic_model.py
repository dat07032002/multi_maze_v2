"""Closed-form ball-on-plate motion, valid between contacts.

Checked against this rig rather than assumed:

* **No slip.** Rolling needs only ``mu >= (2/7) tan(theta)`` -- 0.020 at 4
  degrees, 0.042 at the 8.36 degree hard limit -- against a floor friction of
  0.15-0.7. The ball rolls everywhere in the operating range and never slides.
* **Plate rotation contributes nothing.** The centrifugal and Coriolis terms go
  as ``x * beta_dot^2``: 0.128 m times (8.2 deg/s)^2 is 0.0026 m/s^2, against a
  0.489 m/s^2 gravity term at 4 degrees. 0.54 %, at maximum slew and maximum
  offset from centre. Dropped.

So each axis is an independent double integrator with a single gain::

    x'' = (5/7) g sin(beta)  ~=  7.007 * beta
    y'' = -(5/7) g sin(alpha)

and the full commanded-tilt-to-position plant is
``7.007 * e^(-Ts) / (s^2 (tau s + 1))``. That is what makes this problem hard in
classical terms: the double integrator is already at -180 degrees of phase and
the delay alone adds another 180 by 2.70 Hz, so no gain stabilises it without
prediction.

This module is deliberately *not* a second simulator. It exists for the three
jobs MuJoCo is the wrong tool for: predicting the ball across the dead time,
supplying the process model to the Kalman filter, and giving the baseline
controller something to invert. Wall impacts and hole capture stay with MuJoCo.
"""
from __future__ import annotations

import math

import numpy as np

GRAVITY = 9.81
ROLLING_GAIN = 5.0 / 7.0
#: In-plane acceleration per radian of tilt, for small angles: 7.007 m/s^2/rad.
ACCEL_PER_RAD = ROLLING_GAIN * GRAVITY


#: Lumped linear damping, 1/s. Fitted at M2 against the MuJoCo model as shipped
#: (see ``ball.linear_damping`` in parameters.json). Without it the analytic
#: model overshoots the simulator by 7.8 % at 0.4 degrees and 2.2 % at 4 -- worst
#: when the ball is slow, which is exactly where a controller is trying to hold
#: it still.
DEFAULT_DAMPING = 0.2018

#: Constant-force rolling resistance, m/s^2. The physically correct companion to
#: the linear term: MuJoCo's condim=6 rolling friction opposes motion with a
#: roughly velocity-INDEPENDENT force, which velocity-proportional damping alone
#: cannot represent -- the mismatch peaks at mid tilt, where the ball is neither
#: slow (damping dominates) nor fast. Fitted alongside ``ball.linear_damping``
#: against the shipped MuJoCo model. Zero by default so the pure law stays
#: available for the physics checks.
DEFAULT_COULOMB = 0.0

#: Below this speed the Coulomb term is not applied, so a near-stationary ball
#: cannot be pushed backwards by its own rolling resistance.
_COULOMB_SPEED_FLOOR = 1e-4


def acceleration(alpha: float, beta: float, velocity=None,
                 damping: float = 0.0, coulomb: float = 0.0) -> np.ndarray:
    """Board-frame in-plane acceleration for a tilt of ``(alpha, beta)``.

    Uses ``sin`` rather than the small-angle form. It costs nothing and the
    difference at the 8.36 degree travel limit is 0.4 %, which is the same order
    as the parameters this feeds.

    Two resistance terms, both opposing motion and both zero by default:
    ``damping`` scales with speed (contact-solver residual) and ``coulomb`` is a
    constant deceleration (rolling resistance). Together they stand in for the
    simulator's coast-down; see ``ball.linear_damping`` and
    ``ball.rolling_coulomb`` in parameters.json.
    """
    accel = np.array([ACCEL_PER_RAD * math.sin(beta),
                      -ACCEL_PER_RAD * math.sin(alpha)])
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=float)
        if damping:
            accel = accel - damping * velocity
        if coulomb:
            speed = float(np.linalg.norm(velocity))
            if speed > _COULOMB_SPEED_FLOOR:
                accel = accel - coulomb * (velocity / speed)
    return accel


def step(position, velocity, alpha: float, beta: float, dt: float,
         damping: float = 0.0, coulomb: float = 0.0):
    """Advance one timestep under constant tilt, integrated exactly.

    Constant acceleration over the interval has a closed form, so there is no
    reason to use an Euler step and inherit its error.
    """
    velocity = np.asarray(velocity, dtype=float)
    accel = acceleration(alpha, beta, velocity, damping, coulomb)
    position = np.asarray(position, dtype=float) + \
        velocity * dt + 0.5 * accel * dt * dt
    return position, velocity + accel * dt


def rollout(position, velocity, angles, dt: float, damping: float = 0.0,
            coulomb: float = 0.0):
    """Integrate through a sequence of ``(alpha, beta)`` pairs."""
    for alpha, beta in angles:
        position, velocity = step(
            position, velocity, alpha, beta, dt, damping, coulomb)
    return position, velocity


def min_time_to_go(distance: float, speed: float, max_tilt: float) -> float:
    """Minimum time to cover ``distance`` from ``speed`` and stop, bang-bang.

    The textbook double-integrator answer. Used as a *diagnostic* -- it says how
    long a stretch of route ought to take, which separates "the policy is slow"
    from "this section is genuinely hard". Deliberately not used as reward
    shaping; see the reward section of the plan.
    """
    accel = ACCEL_PER_RAD * math.sin(max_tilt)
    if accel <= 0.0:
        return math.inf
    # Accelerate to a peak then brake to rest, symmetric about the midpoint.
    peak = math.sqrt(max(0.0, speed * speed / 2.0 + accel * distance))
    return (2.0 * peak - speed) / accel


def switching_curve(distance: float, speed: float, accel: float) -> float:
    """Sign of the time-optimal command for a double integrator.

    Positive means drive toward the target, negative means brake. The curve is
    ``distance - speed*|speed|/(2a)``: on it, releasing now arrives exactly at
    rest. This is what the baseline controller inverts.
    """
    return distance - speed * abs(speed) / (2.0 * accel)
