"""Sampling-based MPC teacher for local maze maneuvers.

The optimizer uses a vectorized control-rate surrogate of the measured plant:
per-axis dead time, first-order lag, slew limits and linkage backlash feed the
closed-form rolling-ball dynamics.  MuJoCo remains the executed plant, so the
teacher replans after every real simulator transition and contacts are never
trusted to the between-contact analytical model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from contract import policy_contract as pc
from control.baseline import PurePursuitBaseline
from sim.analytic_model import ACCEL_PER_RAD


@dataclass(frozen=True)
class CEMConfig:
    horizon_steps: int = 30       # 1.5 s at the 20 Hz control rate
    candidates: int = 192
    iterations: int = 4
    elites: int = 24
    initial_std: float = 0.28
    minimum_std: float = 0.035
    noise_correlation: float = 0.72
    # The surrogate intentionally omits contacts, so large optimizer residuals
    # can turn predicted improvements into real wall stalls.  Keep MPC as a
    # small correction to the measured-physics controller.
    max_residual_action: float = 0.03


class CEMMPCTeacher:
    """Receding-horizon teacher returning normalized policy actions."""

    def __init__(self, env, seed: int = 0, config: CEMConfig | None = None):
        self.env = env
        self.config = config or CEMConfig()
        if not 0 < self.config.elites < self.config.candidates:
            raise ValueError("CEM elites must lie between zero and candidates")
        self.rng = np.random.default_rng(seed)
        self.route = env.route
        self.params = env._params
        self.max_tilt = env.max_tilt
        self.dt = 1.0 / env.control_hz
        self.baseline = PurePursuitBaseline(
            self.route, self.max_tilt,
            speed_max=max(0.012, env.desired_speed))
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def reset(self) -> None:
        self._mean = None
        self._std = None

    def _state(self):
        position, velocity = self.env.estimator.state
        position, velocity = self.env.predictor.predict(
            position, velocity, self.env._command)
        return np.asarray(position), np.asarray(velocity)

    def _baseline_action(self, position, velocity) -> np.ndarray:
        angles = self.baseline(position, velocity)
        return np.asarray(pc.angles_to_action(
            *angles, self.max_tilt), dtype=float)

    def _sample(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        cfg = self.config
        noise = self.rng.normal(size=(cfg.candidates, cfg.horizon_steps, 2))
        rho = cfg.noise_correlation
        scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        for step in range(1, cfg.horizon_steps):
            noise[:, step] = rho * noise[:, step - 1] + scale * noise[:, step]
        return np.clip(mean[None] + std[None] * noise, -1.0, 1.0)

    def _initial_actuator_state(self, count: int):
        shafts, plates, delays, taus, rates, backlash = [], [], [], [], [], []
        for name in ("roll", "pitch"):
            plant = self.env.actuator.plants[name]
            shafts.append(plant.shaft_rad)
            plates.append(plant.plate_rad)
            delays.append(max(0, int(round(plant.dyn.dead_time_s / self.dt))))
            taus.append(plant.dyn.tau_s)
            rates.append(plant.dyn.max_rate_rad_s)
            backlash.append(plant.dyn.backlash_rad)
        return (
            np.broadcast_to(np.asarray(shafts), (count, 2)).copy(),
            np.broadcast_to(np.asarray(plates), (count, 2)).copy(),
            np.asarray(delays, dtype=int), np.asarray(taus),
            np.asarray(rates), np.asarray(backlash),
        )

    def _score(self, action_sequences: np.ndarray,
               position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Roll all candidate sequences through the analytical surrogate."""
        count, horizon, _ = action_sequences.shape
        positions = np.broadcast_to(position, (count, 2)).copy()
        velocities = np.broadcast_to(velocity, (count, 2)).copy()
        shaft, plate, delays, taus, rates, backlash = \
            self._initial_actuator_state(count)

        max_delay = int(max(delays.max(), 1))
        history = np.asarray(self.env._history, dtype=float)
        if len(history) < max_delay:
            history = np.vstack([
                np.repeat(history[:1], max_delay - len(history), axis=0),
                history,
            ])
        else:
            history = history[-max_delay:]
        history_angles = history * self.max_tilt
        commands = action_sequences * self.max_tilt
        timeline = np.concatenate([
            np.broadcast_to(history_angles, (count, max_delay, 2)), commands
        ], axis=1)

        route_points = self.route.points
        route_s = self.route.s
        route_tangents = self.route.tangents
        route_clearance = np.asarray(self.route.clearance)
        route_clearance = np.where(
            np.isfinite(route_clearance), route_clearance,
            float(self.env.geometry["min_clearance_m"]))
        nearest = np.argmin(
            np.linalg.norm(route_points[None] - positions[:, None], axis=2),
            axis=1)
        initial_s = route_s[nearest]
        furthest_s = initial_s.copy()
        previous_action = np.broadcast_to(
            np.asarray(self.env._last_action), (count, 2)).copy()
        costs = np.zeros(count)
        board_low = self.env.ball_radius
        board_high = self.env.board_size - self.env.ball_radius
        damping = self.params["ball.linear_damping"]
        coulomb = self.params.get("ball.rolling_coulomb", 0.0)
        blend = 1.0 - np.exp(-self.dt / np.maximum(taus, 1e-6))

        for step in range(horizon):
            delayed = np.empty((count, 2))
            for axis in range(2):
                delayed[:, axis] = timeline[
                    :, max_delay + step - delays[axis], axis]
            shaft_move = np.clip(
                (delayed - shaft) * blend,
                -rates * self.dt, rates * self.dt)
            shaft += shaft_move
            plate = np.minimum(np.maximum(plate, shaft), shaft + backlash)

            accel = np.column_stack([
                ACCEL_PER_RAD * np.sin(plate[:, 1]),
                -ACCEL_PER_RAD * np.sin(plate[:, 0]),
            ]) - damping * velocities
            if coulomb:
                speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
                accel -= coulomb * np.divide(
                    velocities, speeds, out=np.zeros_like(velocities),
                    where=speeds > 1e-4)
            positions += velocities * self.dt + 0.5 * accel * self.dt**2
            velocities += accel * self.dt

            distances = np.linalg.norm(
                route_points[None] - positions[:, None], axis=2)
            nearest = np.argmin(distances, axis=1)
            cross = distances[np.arange(count), nearest]
            progress_s = route_s[nearest]
            furthest_s = np.maximum(furthest_s, progress_s)
            clearance = np.maximum(route_clearance[nearest], 0.001)
            risk = cross / clearance
            desired_velocity = route_tangents[nearest] * self.env.desired_speed
            velocity_error = np.linalg.norm(
                velocities - desired_velocity, axis=1) / max(
                    self.env.desired_speed, 0.005)
            action_change = np.linalg.norm(
                action_sequences[:, step] - previous_action, axis=1)
            step_progress = np.maximum(0.0, progress_s - initial_s) \
                / max(self.route.length, 1e-6)
            costs += (1.2 * np.square(risk)
                      + 0.16 * np.square(velocity_error)
                      + 0.025 * np.square(action_change)
                      - 0.45 * step_progress)
            outside = np.any((positions < board_low) |
                             (positions > board_high), axis=1)
            unsafe = (risk > 0.92) | outside
            costs += unsafe * (250.0 + 20.0 * np.square(risk))
            previous_action = action_sequences[:, step]

        completion = np.clip(
            furthest_s / max(self.route.length, 1e-6), 0.0, 1.0)
        final_tangent = route_tangents[nearest]
        lateral_velocity = np.abs(
            velocities[:, 0] * -final_tangent[:, 1]
            + velocities[:, 1] * final_tangent[:, 0])
        costs += 55.0 * (1.0 - completion) \
            + 180.0 * np.square(lateral_velocity)
        return costs

    def __call__(self, _observation=None) -> np.ndarray:
        cfg = self.config
        position, velocity = self._state()
        baseline = self._baseline_action(position, velocity)
        if self._mean is None:
            mean = np.broadcast_to(baseline, (cfg.horizon_steps, 2)).copy()
            std = np.full_like(mean, cfg.initial_std)
        else:
            mean = np.vstack([self._mean[1:], self._mean[-1:]])
            std = np.vstack([self._std[1:], self._std[-1:]])
            # Retain the analytic controller as a conservative prior rather
            # than letting a noisy finite-sample optimizer drift arbitrarily.
            mean = 0.75 * mean + 0.25 * baseline

        best_sequence = np.broadcast_to(
            baseline, (cfg.horizon_steps, 2)).copy()
        for _ in range(cfg.iterations):
            samples = self._sample(mean, std)
            samples[0] = np.broadcast_to(
                baseline, (cfg.horizon_steps, 2))
            scores = self._score(samples, position, velocity)
            best_sequence = samples[int(np.argmin(scores))].copy()
            elite = samples[np.argpartition(scores, cfg.elites)[:cfg.elites]]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), cfg.minimum_std)

        self._mean, self._std = mean, std
        # CEM's distribution mean can lie between distinct elite modes and be
        # worse than every member (particularly left-vs-right braking modes).
        # Execute the best sampled sequence, bounded as a residual around the
        # already competent analytic controller.  This makes MPC an improving
        # teacher rather than allowing finite-sample optimization to erase a
        # known-safe feedback law.
        residual = np.clip(best_sequence[0] - baseline,
                           -cfg.max_residual_action,
                           cfg.max_residual_action)
        return np.clip(baseline + residual, -1.0, 1.0).astype(np.float32)
