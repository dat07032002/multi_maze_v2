"""Full-maze observation, reward, and termination logic."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ObservationSpec:
    """Stable named layout shared by replay, dynamics, CEM, and dashboards."""

    ray_count: int = 12

    @property
    def names(self) -> tuple[str, ...]:
        base = (
            "x", "y", "vx", "vy", "alpha", "beta", "alpha_rate",
            "beta_rate", "previous_alpha", "previous_beta", "progress",
            "cross_track", "target_dx", "target_dy", "tangent_x",
            "tangent_y", "clearance", "stuck",
        )
        return base + tuple(f"ray_{index}" for index in range(self.ray_count))

    @property
    def size(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass(frozen=True)
class RouteState:
    progress_mm: float
    cross_track_mm: float
    target_vector_mm: np.ndarray
    tangent: np.ndarray
    route_index: int
    distance_to_goal_mm: float


@dataclass(frozen=True)
class TaskStep:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    reason: str | None
    route: RouteState


class MazeTask:
    """Map-aware task for one camera-planned full route.

    Physical units are retained deliberately. Neural normalization belongs to
    the model, not the environment or audit log.
    """

    def __init__(self, route_points_mm, occupied: np.ndarray, *,
                 board_size_mm=(256.0, 226.0), resolution_mm: float = 1.0,
                 ray_count: int = 12, ray_length_mm: float = 30.0,
                 lookahead_mm: float = 20.0, goal_radius_mm: float = 7.0,
                 episode_timeout_s: float = 60.0) -> None:
        route = np.asarray(route_points_mm, dtype=np.float64)
        occupied = np.asarray(occupied, dtype=bool)
        if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
            raise ValueError("route must be Nx2")
        if occupied.ndim != 2 or resolution_mm <= 0:
            raise ValueError("invalid occupancy map")
        self.route_points = route
        self.occupied = occupied
        self._clearance_map = cv2.distanceTransform(
            (~self.occupied).astype(np.uint8), cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE) * float(resolution_mm)
        self.board_width_mm, self.board_height_mm = map(float, board_size_mm)
        self.resolution_mm = float(resolution_mm)
        self.spec = ObservationSpec(int(ray_count))
        self.ray_length_mm = float(ray_length_mm)
        self.lookahead_mm = float(lookahead_mm)
        self.goal_radius_mm = float(goal_radius_mm)
        self.episode_timeout_s = float(episode_timeout_s)
        segment = np.diff(route, axis=0)
        self.segment_lengths = np.linalg.norm(segment, axis=1)
        if np.any(self.segment_lengths <= 1e-9):
            raise ValueError("route contains duplicate consecutive points")
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.route_length_mm = float(self.cumulative[-1])
        self._route_index = 0
        self._previous_progress = 0.0
        self._episode_start: float | None = None

    @classmethod
    def load(cls, map_json: str | Path, occupied_png: str | Path,
             **kwargs) -> "MazeTask":
        metadata = json.loads(Path(map_json).read_text(encoding="utf-8"))
        if not metadata.get("route"):
            raise ValueError("map JSON contains no route")
        image = cv2.imread(str(occupied_png), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read occupancy map {occupied_png}")
        return cls(
            metadata["route"]["points_mm"], image > 127,
            board_size_mm=metadata["board_size_mm"],
            resolution_mm=metadata["resolution_mm"], **kwargs)

    def reset(self, timestamp_s: float, position_mm=None) -> None:
        self._route_index = 0
        if position_mm is not None:
            position = np.asarray(position_mm, dtype=np.float64)
            if position.shape != (2,) or not np.all(np.isfinite(position)):
                raise ValueError("position_mm must be a finite two-vector")
            # Reloads need not land at the original route entrance. Project
            # globally once, then resume the bounded monotonic tracking window.
            candidates = [self._project_segment(position, index)
                          for index in range(len(self.segment_lengths))]
            self._route_index = int(np.argmin(
                [candidate[0] for candidate in candidates]))
            progress = candidates[self._route_index][1]
        else:
            progress = 0.0
        self._previous_progress = float(progress)
        self._episode_start = float(timestamp_s)

    def _project_segment(self, position: np.ndarray, index: int):
        a, b = self.route_points[index:index + 2]
        vector = b - a
        fraction = float(np.clip(np.dot(position - a, vector)
                                 / np.dot(vector, vector), 0.0, 1.0))
        projected = a + fraction * vector
        tangent = vector / self.segment_lengths[index]
        delta = position - projected
        cross = float(tangent[0] * delta[1] - tangent[1] * delta[0])
        progress = self.cumulative[index] + fraction * self.segment_lengths[index]
        return float(np.dot(delta, delta)), progress, cross, projected, tangent

    def route_state(self, position_mm) -> RouteState:
        position = np.asarray(position_mm, dtype=np.float64)
        if position.shape != (2,):
            raise ValueError("position must be a two-vector")
        lo = max(0, self._route_index - 3)
        hi = min(len(self.segment_lengths), self._route_index + 24)
        candidates = [self._project_segment(position, index)
                      for index in range(lo, hi)]
        local = int(np.argmin([item[0] for item in candidates]))
        index = lo + local
        _, progress, cross, _, tangent = candidates[local]
        self._route_index = max(self._route_index, index)
        target_progress = min(self.route_length_mm, progress + self.lookahead_mm)
        target = self._point_at_progress(target_progress)
        return RouteState(
            progress_mm=progress, cross_track_mm=cross,
            target_vector_mm=target - position, tangent=tangent.copy(),
            route_index=index,
            distance_to_goal_mm=float(np.linalg.norm(
                position - self.route_points[-1])))

    def _point_at_progress(self, progress_mm: float) -> np.ndarray:
        progress = float(np.clip(progress_mm, 0.0, self.route_length_mm))
        index = min(int(np.searchsorted(self.cumulative, progress, side="right") - 1),
                    len(self.segment_lengths) - 1)
        fraction = ((progress - self.cumulative[index])
                    / self.segment_lengths[index])
        return (self.route_points[index] + fraction
                * (self.route_points[index + 1] - self.route_points[index]))

    def board_to_grid(self, position_mm) -> tuple[int, int]:
        x, y = map(float, position_mm)
        col = int(round(x / self.resolution_mm))
        row = int(round((self.board_height_mm - y) / self.resolution_mm))
        return row, col

    def clearance_rays(self, position_mm) -> tuple[float, np.ndarray]:
        row, col = self.board_to_grid(position_mm)
        if not (0 <= row < self.occupied.shape[0]
                and 0 <= col < self.occupied.shape[1]):
            return 0.0, np.zeros(self.spec.ray_count)
        # The occupancy map is immutable, so its exact distance transform is
        # precomputed once rather than rebuilt for every camera observation.
        clearance = (0.0 if self.occupied[row, col]
                     else float(self._clearance_map[row, col]))
        rays = np.empty(self.spec.ray_count, dtype=np.float64)
        samples = np.arange(self.resolution_mm, self.ray_length_mm
                            + self.resolution_mm, self.resolution_mm)
        x, y = map(float, position_mm)
        for index, angle in enumerate(np.linspace(
                0.0, 2.0 * math.pi, self.spec.ray_count, endpoint=False)):
            rays[index] = self.ray_length_mm
            for radius in samples:
                sample = (x + radius * math.cos(angle),
                          y + radius * math.sin(angle))
                sr, sc = self.board_to_grid(sample)
                if (not 0 <= sr < self.occupied.shape[0]
                        or not 0 <= sc < self.occupied.shape[1]
                        or self.occupied[sr, sc]):
                    rays[index] = float(radius)
                    break
        return clearance, rays

    def observation(self, *, position_mm, velocity_mm_s, angles_deg,
                    angle_rates_deg_s, previous_action_deg,
                    stuck: bool = False) -> tuple[np.ndarray, RouteState]:
        position = np.asarray(position_mm, dtype=np.float64)
        velocity = np.asarray(velocity_mm_s, dtype=np.float64)
        angles = np.asarray(angles_deg, dtype=np.float64)
        rates = np.asarray(angle_rates_deg_s, dtype=np.float64)
        previous = np.asarray(previous_action_deg, dtype=np.float64)
        for value in (position, velocity, angles, rates, previous):
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError("all physical observation fields must be finite 2-vectors")
        route = self.route_state(position)
        clearance, rays = self.clearance_rays(position)
        values = np.concatenate((
            position, velocity, angles, rates, previous,
            [route.progress_mm, route.cross_track_mm],
            route.target_vector_mm, route.tangent,
            [clearance, float(bool(stuck))], rays,
        )).astype(np.float32)
        if values.shape != (self.spec.size,):
            raise RuntimeError("observation layout mismatch")
        return values, route

    def step_result(self, observation: np.ndarray, route: RouteState, *,
                    timestamp_s: float, ball_lost: bool = False,
                    safety_abort: bool = False) -> TaskStep:
        values = np.asarray(observation, dtype=np.float32)
        speed = float(np.linalg.norm(values[[self.spec.index("vx"),
                                             self.spec.index("vy")]]))
        clearance = float(values[self.spec.index("clearance")])
        progress_delta = route.progress_mm - self._previous_progress
        self._previous_progress = max(self._previous_progress, route.progress_mm)
        reward = progress_delta / 5.0
        reward -= 0.015 * abs(route.cross_track_mm)
        reward -= 0.002 * max(0.0, 12.0 - clearance) ** 2
        if clearance < 12.0:
            reward -= 0.00015 * speed ** 2
        reason = None
        terminated = False
        truncated = False
        if route.distance_to_goal_mm <= self.goal_radius_mm:
            reward += 25.0
            terminated, reason = True, "goal"
        elif safety_abort:
            reward -= 15.0
            terminated, reason = True, "safety_abort"
        elif ball_lost:
            reward -= 12.0
            terminated, reason = True, "ball_lost"
        elif (self._episode_start is not None
              and timestamp_s - self._episode_start >= self.episode_timeout_s):
            reward -= 3.0
            truncated, reason = True, "timeout"
        return TaskStep(values, float(reward), terminated, truncated, reason, route)
