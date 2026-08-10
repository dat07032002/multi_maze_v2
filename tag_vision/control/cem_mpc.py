"""Clearance-aware CEM model-predictive controller for a planned route."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ball_dynamics import BallDynamicsModel


@dataclass(frozen=True)
class MPCResult:
    tilt_deg: np.ndarray
    predicted_states: np.ndarray
    cost: float


class RouteMPC:
    """Optimize short tilt sequences against route error and occupancy.

    This is intentionally dependency-free NumPy so it can run on the laptop.
    Five GPUs are useful later for replay updates, not for this 2-D optimizer.
    """

    def __init__(self, model: BallDynamicsModel, route_points_mm,
                 occupied: np.ndarray, *, resolution_mm: float = 1.0,
                 board_height_mm: float = 226.0, dt_s: float = 0.20,
                 horizon: int = 10, candidates: int = 384,
                 iterations: int = 4, elite_fraction: float = 0.10,
                 max_tilt_deg: float = 3.0, seed: int | None = None) -> None:
        route = np.asarray(route_points_mm, dtype=np.float64)
        occupied = np.asarray(occupied, dtype=bool)
        if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
            raise ValueError("route_points_mm must be Nx2 with N >= 2")
        if occupied.ndim != 2 or resolution_mm <= 0 or dt_s <= 0:
            raise ValueError("invalid occupancy or timing")
        if horizon < 2 or candidates < 8 or iterations < 1:
            raise ValueError("CEM dimensions are too small")
        self.model = model
        self.route = route
        self.occupied = occupied
        self.resolution_mm = float(resolution_mm)
        self.board_height_mm = float(board_height_mm)
        self.dt_s = float(dt_s)
        self.horizon = int(horizon)
        self.candidates = int(candidates)
        self.iterations = int(iterations)
        self.elites = max(2, int(round(candidates * elite_fraction)))
        self.max_tilt_deg = float(max_tilt_deg)
        self.rng = np.random.default_rng(seed)
        self._mean = np.zeros((self.horizon, 2), dtype=np.float64)
        self._progress_index = 0

    def reset(self) -> None:
        self._mean[:] = 0.0
        self._progress_index = 0

    def _route_index(self, xy: np.ndarray) -> int:
        # Permit a small backward correction but prevent nearest-point jumps to
        # a nearby later corridor on maze switchbacks.
        lo = max(0, self._progress_index - 3)
        hi = min(len(self.route), self._progress_index + 24)
        index = lo + int(np.argmin(np.sum((self.route[lo:hi] - xy) ** 2, axis=1)))
        self._progress_index = max(self._progress_index, index)
        return index

    def _is_occupied(self, positions: np.ndarray) -> np.ndarray:
        columns = np.rint(positions[..., 0] / self.resolution_mm).astype(int)
        rows = np.rint(
            (self.board_height_mm - positions[..., 1]) / self.resolution_mm
        ).astype(int)
        outside = ((rows < 0) | (rows >= self.occupied.shape[0])
                   | (columns < 0) | (columns >= self.occupied.shape[1]))
        rows = np.clip(rows, 0, self.occupied.shape[0] - 1)
        columns = np.clip(columns, 0, self.occupied.shape[1] - 1)
        return outside | self.occupied[rows, columns]

    def _rollout(self, initial: np.ndarray, actions: np.ndarray) -> np.ndarray:
        count = len(actions)
        states = np.empty((count, self.horizon + 1, 4), dtype=np.float64)
        states[:, 0] = initial
        b = self.model.acceleration_per_tilt
        d = self.model.velocity_damping
        bias = self.model.bias_mm_s2
        dt = self.dt_s
        for step in range(self.horizon):
            current = states[:, step]
            acceleration = actions[:, step] @ b.T - current[:, 2:] @ d.T + bias
            states[:, step + 1, :2] = (
                current[:, :2] + current[:, 2:] * dt
                + 0.5 * acceleration * dt * dt)
            states[:, step + 1, 2:] = current[:, 2:] + acceleration * dt
        return states

    def _cost(self, states: np.ndarray, actions: np.ndarray,
              route_index: int) -> np.ndarray:
        # A moving reference rewards forward progress without allowing shortcut
        # jumps through a neighbouring segment of the route.
        offsets = np.arange(1, self.horizon + 1) * 2
        indices = np.minimum(route_index + offsets, len(self.route) - 1)
        references = self.route[indices]
        error2 = np.sum((states[:, 1:, :2] - references[None, :, :]) ** 2,
                        axis=2)
        cost = 0.035 * np.sum(error2, axis=1)
        cost += 0.0015 * np.sum(states[:, 1:, 2:] ** 2, axis=(1, 2))
        cost += 0.20 * np.sum(actions ** 2, axis=(1, 2))
        changes = np.diff(actions, axis=1)
        cost += 0.35 * np.sum(changes ** 2, axis=(1, 2))
        collision = self._is_occupied(states[:, 1:, :2])
        cost += 1_000_000.0 * np.any(collision, axis=1)
        goal_error2 = np.sum((states[:, -1, :2] - references[-1]) ** 2,
                             axis=1)
        return cost + 0.15 * goal_error2

    def command(self, state) -> MPCResult:
        initial = np.asarray(state, dtype=np.float64)
        if initial.shape != (4,) or not np.all(np.isfinite(initial)):
            raise ValueError("state must be finite [x,y,vx,vy]")
        route_index = self._route_index(initial[:2])
        mean = np.vstack((self._mean[1:], self._mean[-1:]))
        std = np.full_like(mean, 1.25)
        best_actions = mean.copy()
        best_states = self._rollout(initial, best_actions[None])[0]
        best_cost = float("inf")
        for _ in range(self.iterations):
            actions = self.rng.normal(mean, std,
                                      size=(self.candidates, self.horizon, 2))
            actions = np.clip(actions, -self.max_tilt_deg, self.max_tilt_deg)
            actions[0] = np.clip(mean, -self.max_tilt_deg, self.max_tilt_deg)
            states = self._rollout(initial, actions)
            costs = self._cost(states, actions, route_index)
            elite_indices = np.argpartition(costs, self.elites)[:self.elites]
            elites = actions[elite_indices]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), 0.08)
            winner = int(np.argmin(costs))
            if float(costs[winner]) < best_cost:
                best_cost = float(costs[winner])
                best_actions = actions[winner].copy()
                best_states = states[winner].copy()
        self._mean = mean
        return MPCResult(best_actions[0], best_states, best_cost)
