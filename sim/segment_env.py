"""Local maneuver episodes built from the balanced segment dataset."""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from contract import policy_contract as pc
from sim.maze_env import MazeEnv
from sim.mjcf_builder import flat_board_layout, load_layout, load_parameters
from sim.randomization import Randomizer
from sim.symmetry import mirror_layout


def layout_for_geometry(geometry: dict) -> dict:
    """Return the physical layout and local route described by ``geometry``."""
    base = load_layout()
    variant = geometry["layout_variant"]
    if variant == "original":
        layout = copy.deepcopy(base)
    elif variant == "mirror_x":
        layout = mirror_layout(base, axis=0)
    elif variant == "flat_procedural":
        layout = flat_board_layout(base)
    else:
        raise ValueError(f"unknown layout variant {variant!r}")

    points = geometry["points_m"]
    if len(points) < 2:
        raise ValueError("a segment needs at least two route points")
    layout["waypoints"] = points
    layout["start_planned"] = points[0]
    layout["goal_planned"] = points[-1]
    return layout


class SegmentEnv(MazeEnv):
    """A short route episode with dataset-controlled initial conditions."""

    def __init__(self, geometry: dict, randomization_scale: float = 0.0,
                 max_seconds: float = 60.0, seed: int | None = None,
                 sensor_noise: bool = True):
        self.geometry = geometry
        layout = layout_for_geometry(geometry)
        randomizer = Randomizer(scale=randomization_scale,
                                enabled=randomization_scale > 0.0)
        super().__init__(layout=layout, params=load_parameters(),
                         max_seconds=max_seconds, sensor_noise=sensor_noise,
                         randomizer=randomizer, start_fraction=0.0, seed=seed)

    def _build(self) -> None:
        """Reapply synthetic corridor clearance after every randomized build."""
        super()._build()
        if self.geometry["layout_variant"] == "flat_procedural":
            self.route.clearance[:] = self.geometry["min_clearance_m"]

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        specification = (options or {}).get("episode_spec")
        if specification is None:
            return observation, info

        angles = np.asarray(
            specification["initial_board_angles_rad"], dtype=float)
        position = np.asarray(specification["initial_position_m"], dtype=float)
        velocity = np.asarray(
            specification["initial_velocity_m_s"], dtype=float)

        self.board.set_tilt(self.data, angles[0], angles[1])
        mujoco.mj_forward(self.model, self.data)
        self.board.set_ball(
            self.data, position[0] - self.board_size[0] / 2.0,
            position[1] - self.board_size[1] / 2.0)
        self.board.set_ball_velocity_board(
            self.data, velocity[0], velocity[1])
        mujoco.mj_forward(self.model, self.data)

        self.actuator.reset(angles[0], angles[1])
        self.estimator.reset(position, velocity)
        self._command = (float(angles[0]), float(angles[1]))
        history = np.asarray(
            specification.get("initial_action_history",
                              np.zeros((pc.ACTION_HISTORY, 2))), dtype=float)
        self._history = [row.copy() for row in history]
        self._last_action = self._history[-1].copy()
        self._reading_delay = []
        self._steps = 0
        self._max_s = 0.0
        self._cross_track = []
        self.max_steps = int(round(self.max_seconds * self.control_hz))
        return self._observe(), self._info(0.0)

    @property
    def desired_speed(self) -> float:
        # Raised 2026-08-08 from 25/20/12. Corner-speed physics caps are
        # ~77 mm/s (sharp, r~12mm) to ~106 mm/s (gentle, r~23mm) at the 4 deg
        # authority, so 12-20 was far under the physical limit -- the real
        # constraint is tracking precision under dead time, not the corner. This
        # lifts the teacher's floor; the residual + time cost push past it.
        kind = self.geometry["kind"]
        if kind == "straight":
            return 0.032
        if kind.startswith("gentle"):
            return 0.026
        return 0.015
