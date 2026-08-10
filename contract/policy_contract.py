"""What the policy sees and what it emits. Shared by simulation and the rig.

Sits beside ``servo_contract.py`` and above it: this turns board state into an
observation vector, that turns an action into servo counts. Neither knows about
the other, which is the point -- a change to the linkage cannot alter the policy
interface and a change to the observation cannot alter the servo mapping.

Three decisions are baked in here, all of which have consequences elsewhere.

**State, not pixels.** The perception stack already reports the marble in metres.
Handing a policy images instead would ask a world model to relearn physics that
``sim/analytic_model.py`` writes down in closed form.

**The ball state is predicted, not measured.** The camera describes the past by
its latency and the actuator will not respond for its dead time, so the state
that matters is the one about 240 ms ahead. Feeding the measurement instead
makes the problem non-Markov and forces the network to infer the delay from a
long action history. With prediction, three past actions suffice; without it,
ten were not obviously enough.

**Route lookahead is relative to the ball, not absolute.** That is what makes
the mirror symmetries an exact data augmentation, and it is what will let a
policy trained on this maze mean anything on another one -- the policy never
sees where it is on *this* board, only the shape of the path in front of it.
"""
from __future__ import annotations

import math

import numpy as np

CONTRACT_VERSION = "tag_maze_policy_v2"

#: Velocity normaliser. 0.50 m/s is what the ball reaches crossing the board at
#: the +-4 degree command limit, so the observation sits inside [-1, 1] in
#: ordinary play without clipping away information at the extremes.
REFERENCE_SPEED = 0.50

#: Arc-length spacing of the lookahead points, and how many.
LOOKAHEAD_SPACING = 0.012
LOOKAHEAD_COUNT = 5

#: Past actions retained. Covers 150 ms at 20 Hz, on top of the predictor.
ACTION_HISTORY = 3

#: Clearance normaliser, metres. Route clearance runs from about -2 mm (the
#: centreline inside a dodge's widened cell) to 30 mm (an open cell), so a
#: 10 mm reference puts a tight dodge near 0.15, a normal corridor near 1.0 and
#: an open cell at the 2.0 clip. This is the signal the policy keys off to crawl
#: at a squeeze and hurry in the open; it is a precomputed property of the known
#: map, a table lookup at deployment, not something perception has to recover.
LOOKAHEAD_CLEARANCE_REF = 0.010
_CLEARANCE_CLIP = (-1.0, 2.0)

OBSERVATION_SIZE = (
    2 + 2 + 2 + 2 * ACTION_HISTORY + 2 * LOOKAHEAD_COUNT + LOOKAHEAD_COUNT)  # 27


def observation(ball_xy, ball_velocity, board_angles, action_history,
                lookahead_xy, lookahead_clearance, board_size,
                max_tilt_rad) -> np.ndarray:
    """Assemble the 27-float observation.

    ``lookahead_xy`` are absolute route points; they are made ball-relative and
    normalised by their own horizon here, so each entry answers "how far off the
    straight-ahead is the path, k steps out" on a comparable scale rather than
    being dominated by the furthest point. ``lookahead_clearance`` is the room
    the ball has at each of those points, normalised by ``LOOKAHEAD_CLEARANCE_REF``
    -- a scalar per point, so it is unchanged under the mirror augmentation.
    """
    ball_xy = np.asarray(ball_xy, dtype=np.float64)
    board_size = np.asarray(board_size, dtype=np.float64)

    horizons = LOOKAHEAD_SPACING * np.arange(1, LOOKAHEAD_COUNT + 1)
    relative = (np.asarray(lookahead_xy, dtype=np.float64) - ball_xy) \
        / horizons[:, None]
    clearance = np.clip(
        np.asarray(lookahead_clearance, dtype=np.float64)
        / LOOKAHEAD_CLEARANCE_REF, *_CLEARANCE_CLIP)

    values = np.concatenate([
        ball_xy / board_size,
        np.asarray(ball_velocity, dtype=np.float64) / REFERENCE_SPEED,
        np.asarray(board_angles, dtype=np.float64) / max_tilt_rad,
        np.asarray(action_history, dtype=np.float64).reshape(-1),
        relative.reshape(-1),
        clearance.reshape(-1),
    ])
    if values.shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"observation is {values.shape}, expected ({OBSERVATION_SIZE},)")
    return values.astype(np.float32)


def validate(observation_vector) -> None:
    values = np.asarray(observation_vector)
    if values.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"observation must be {OBSERVATION_SIZE} floats")
    if not np.all(np.isfinite(values)):
        raise ValueError("observation contains a non-finite value")


def action_to_angles(action, max_tilt_rad: float) -> tuple[float, float]:
    """Policy action in [-1, 1]^2 to commanded board angles.

    Non-finite input is rejected rather than clipped, matching
    ``ServoContract.action_to_angle``: a NaN reaching the servos is a bug worth
    surfacing, not smoothing over.
    """
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (2,):
        raise ValueError(f"action must be 2 values, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"action contains a non-finite value: {values!r}")
    clipped = np.clip(values, -1.0, 1.0) * max_tilt_rad
    return (float(clipped[0]), float(clipped[1]))


def angles_to_action(alpha: float, beta: float, max_tilt_rad: float):
    """Inverse, for replaying a controller's angles as actions."""
    return (float(np.clip(alpha / max_tilt_rad, -1.0, 1.0)),
            float(np.clip(beta / max_tilt_rad, -1.0, 1.0)))


#: One 40-count command is 0.19 deg; over the +-4 deg action range that is
#: 0.0475 in action units. The action-rate cost saturates at two of them.
ACTION_QUANTUM = math.radians(0.19) / math.radians(4.0)
ACTION_RATE_SCALE = 2.0 * ACTION_QUANTUM
