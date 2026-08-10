"""M3 random-setpoint ball-on-plate task.

This is deliberately simpler than the maze without changing the deployment
stack.  The policy still sees the same 22 floats and drives the same delayed,
quantised actuator through the estimator and dead-time predictor.  Five points
along the straight line to the target occupy the route-lookahead slots, so M3
tests the observation contract rather than introducing a disposable one.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from contract import policy_contract as pc
from sim.analytic_model import ACCEL_PER_RAD
from sim.maze_env import FALL_PENALTY, MazeEnv
from sim.mjcf_builder import flat_board_layout

TARGET_RADIUS = 0.005
HOLD_SECONDS = 3.0
REACH_PROGRESS_SCALE = 10.0
HOLD_PROGRESS_SCALE = 6.0
SUCCESS_BONUS = 10.0
ACTION_RATE_WEIGHT = 0.020
DISTANCE_COST_WEIGHT = 0.010
VELOCITY_COST_WEIGHT = 0.010
PROXIMITY_RADIUS = 0.025
PROXIMITY_WEIGHT = 0.020
SAMPLE_MARGIN = 0.025
MIN_START_DISTANCE = 0.060


class BallPlateEnv(MazeEnv):
    """Reach a random target and remain within 5 mm for three seconds."""

    def __init__(self, *args, max_seconds: float = 20.0, **kwargs):
        self.target = np.zeros(2, dtype=float)
        super().__init__(*args, layout=flat_board_layout(),
                         max_seconds=max_seconds, start_fraction=0.0, **kwargs)
        self.hold_steps_required = int(round(HOLD_SECONDS * self.control_hz))

    def _sample_pair(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.full(2, SAMPLE_MARGIN)
        high = self.board_size - SAMPLE_MARGIN
        for _ in range(1000):
            start = self._rng.uniform(low, high)
            target = self._rng.uniform(low, high)
            if np.linalg.norm(target - start) >= MIN_START_DISTANCE:
                return start, target
        raise RuntimeError("could not sample separated ball/target positions")

    def reset(self, *, seed=None, options=None):
        # Let MazeEnv rebuild randomized dynamics and initialise all shared
        # state, then replace its route start with the M3 random pair.
        super().reset(seed=seed, options=options)
        start, self.target = self._sample_pair()
        self.board.set_ball(self.data,
                            start[0] - self.board_size[0] / 2,
                            start[1] - self.board_size[1] / 2)
        mujoco.mj_forward(self.model, self.data)
        self.actuator.reset(0.0, 0.0)
        self.estimator.reset(start, (0.0, 0.0))

        self.max_steps = int(round(self.max_seconds * self.control_hz))
        self._history = [np.zeros(2) for _ in range(pc.ACTION_HISTORY)]
        self._last_action = np.zeros(2)
        self._command = (0.0, 0.0)
        self._steps = 0
        self._reading_delay = []
        self._hold_steps = 0
        self._initial_distance = float(np.linalg.norm(self.target - start))
        self._previous_distance = self._initial_distance
        return self._observe(), self._info(0.0)

    def _lookahead_clearance(self, s: float) -> np.ndarray:
        """The bare plate has no walls or holes: clearance is fully open."""
        return np.full(pc.LOOKAHEAD_COUNT, np.inf)

    def _lookahead(self, position: np.ndarray, s: float) -> np.ndarray:
        delta = self.target - np.asarray(position, dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            return np.repeat(self.target[None, :], pc.LOOKAHEAD_COUNT, axis=0)
        direction = delta / distance
        horizons = pc.LOOKAHEAD_SPACING * np.arange(
            1, pc.LOOKAHEAD_COUNT + 1)
        travelled = np.minimum(horizons, distance)
        return position + travelled[:, None] * direction

    def _reward(self, action, wall_contact):
        ball = self._ball_xy()
        distance = float(np.linalg.norm(ball - self.target))
        progress = REACH_PROGRESS_SCALE * (
            self._previous_distance - distance) / self._initial_distance
        self._previous_distance = distance

        action_rate = float(np.clip(
            np.linalg.norm(action - self._last_action) / pc.ACTION_RATE_SCALE,
            0.0, 1.0))
        proximity = float(np.clip(1.0 - distance / PROXIMITY_RADIUS, 0.0, 1.0))
        distance_cost = float(np.clip(
            distance / self._initial_distance, 0.0, 1.0))
        speed = float(np.linalg.norm(
            self.board.ball_velocity_board(self.data)[:2]))
        settling_cost = proximity * float(np.clip(speed / 0.05, 0.0, 1.0))
        # Delta-distance teaches reaching; a small continuing proximity reward
        # distinguishes settling near the target from merely visiting it.  The
        # stronger six-point hold budget makes all 60 consecutive in-radius
        # steps visible to the critic instead of hiding the requirement behind
        # a single sparse terminal bonus.
        reward = (progress + PROXIMITY_WEIGHT * proximity
                  - ACTION_RATE_WEIGHT * action_rate
                  - DISTANCE_COST_WEIGHT * distance_cost
                  - VELOCITY_COST_WEIGHT * settling_cost)

        if distance <= TARGET_RADIUS:
            self._hold_steps += 1
            reward += HOLD_PROGRESS_SCALE / self.hold_steps_required
        else:
            self._hold_steps = 0

        terminated = truncated = False
        outcome = "running"
        if self._ball_depth() < -self.floor_thickness:
            reward += FALL_PENALTY
            terminated, outcome = True, "fell"
        elif self._hold_steps >= self.hold_steps_required:
            reward += SUCCESS_BONUS
            terminated, outcome = True, "goal"
        elif self._steps >= self.max_steps:
            truncated, outcome = True, "timeout"

        return reward, terminated, truncated, {
            "costs": {"action_rate": action_rate,
                      "distance": distance_cost,
                      "settling": settling_cost,
                      "proximity": proximity},
            "outcome": outcome,
            "target_distance": distance,
        }

    def _info(self, reward, costs=None, outcome="running",
              target_distance=None) -> dict:
        distance = self._previous_distance if target_distance is None \
            and hasattr(self, "_previous_distance") else target_distance
        if distance is None:
            distance = 0.0
        completion = 1.0 - float(distance) / max(
            getattr(self, "_initial_distance", 1.0), 1e-9)
        return {
            "outcome": outcome,
            "target_distance": float(distance),
            "completion": float(np.clip(completion, 0.0, 1.0)),
            "held_steps": getattr(self, "_hold_steps", 0),
            "costs": costs or {},
            "steps": self._steps,
        }


class SetpointBaseline:
    """Braking-limited velocity control for the double-integrator plant."""

    def __init__(self, max_tilt_rad: float, centre_bias=(0.0, 0.0),
                 speed_max: float = 0.020, velocity_gain: float = 4.0):
        self.max_tilt = max_tilt_rad
        self.centre_bias = np.asarray(centre_bias, dtype=float)
        self.speed_max = speed_max
        self.kv = velocity_gain

    def __call__(self, position, velocity, target) -> tuple[float, float]:
        error = np.asarray(target) - np.asarray(position)
        limit = ACCEL_PER_RAD * math.sin(self.max_tilt)
        distance = float(np.linalg.norm(error))
        if distance > 1e-12:
            # Do not request a speed that cannot be braked before the target.
            speed = min(self.speed_max, math.sqrt(2.0 * limit * distance))
            desired_velocity = error * (speed / distance)
        else:
            desired_velocity = np.zeros(2)
        accel = self.kv * (desired_velocity - np.asarray(velocity))
        norm = float(np.linalg.norm(accel))
        if norm > limit:
            accel *= limit / norm
        scaled = np.clip(accel / ACCEL_PER_RAD, -1.0, 1.0)
        beta = math.asin(float(scaled[0]))
        alpha = math.asin(float(-scaled[1]))
        angles = np.array([alpha, beta]) - self.centre_bias
        return tuple(np.clip(angles, -self.max_tilt, self.max_tilt))
