"""Closed-loop runner: MuJoCo + actuator + estimator + predictor + controller.

The full deployment path, minus the camera and the servo bus. Written once here
so the M2 baseline, the M3/M4 environments and the M7 hardware loop all agree on
the order things happen in, which is the part that is easy to get subtly wrong:

    measure (noisy, sometimes dropped) -> Kalman update -> predict across the
    dead time -> controller -> actuator -> board angle -> physics

Sensor noise, dropouts and latency are applied here rather than inside the
estimator, because they are properties of the camera, not of the filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from control.estimator import BallEstimator
from control.predictor import StatePredictor
from sim.actuator import ActuatorModel
from sim.board_state import BoardState
from sim.mjcf_builder import build_mjcf, load_layout, load_parameters


@dataclass
class RolloutResult:
    positions: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    estimates: list = field(default_factory=list)
    fell: bool = False
    reached_goal: bool = False
    steps: int = 0

    @property
    def track(self) -> np.ndarray:
        return np.asarray(self.positions)


def run_closed_loop(controller, layout=None, params=None, seed: int = 0,
                    max_seconds: float = 60.0, control_hz: float = 20.0,
                    start_xy=None, sensor_noise: bool = True,
                    goal_radius: float = 0.008, goal_xy=None):
    """Drive the maze with ``controller(position, velocity) -> (alpha, beta)``."""
    layout = layout if layout is not None else load_layout()
    params = params if params is not None else load_parameters()
    rng = np.random.default_rng(seed)

    model = mujoco.MjModel.from_xml_string(build_mjcf(layout, params))
    data = mujoco.MjData(model)
    board = BoardState(model)
    dt = model.opt.timestep

    actuator = ActuatorModel(dt, params=params)
    estimator = BallEstimator(measurement_std=params["camera.position_noise"])
    predictor = StatePredictor(actuator, dt,
                               sensor_latency_s=params["camera.latency"])

    W, H = layout["board_width"], layout["board_height"]
    start = np.asarray(start_xy if start_xy is not None
                       else layout["start_planned"], dtype=float)
    goal = np.asarray(goal_xy if goal_xy is not None
                      else layout["goal_planned"], dtype=float)

    mujoco.mj_forward(model, data)
    board.set_ball(data, start[0] - W / 2, start[1] - H / 2)
    # Refresh xpos/xmat after changing qpos.  This matters whenever start_xy is
    # not the MJCF's default ball position.
    mujoco.mj_forward(model, data)
    actuator.reset(0.0, 0.0)
    estimator.reset(start, (0.0, 0.0))

    steps_per_control = int(round(1.0 / control_hz / dt))
    command = (0.0, 0.0)
    result = RolloutResult()
    noise = params["camera.position_noise"] if sensor_noise else 0.0
    dropout = params["camera.dropout_rate"] if sensor_noise else 0.0

    # The camera reports the past. Buffering the reading is not decoration: the
    # predictor is told to compensate for this latency, so if the measurement
    # arrives instantly the loop over-predicts by exactly the amount it was
    # asked to correct for, and removing the latency makes tracking worse
    # instead of better.
    latency_ticks = int(round(params["camera.latency"] * control_hz))
    reading_delay: list = []

    for tick in range(int(max_seconds * control_hz)):
        x, y, z = board.ball_board(data)
        measured_xy = np.array([x + W / 2, y + H / 2])
        result.positions.append(measured_xy.copy())

        if z < -params["maze.floor_thickness"]:
            result.fell = True
            break
        if np.linalg.norm(measured_xy - goal) < goal_radius:
            result.reached_goal = True
            break

        fresh = None
        if rng.random() >= dropout:
            fresh = measured_xy + rng.normal(0.0, noise, size=2) if noise \
                else measured_xy.copy()
        reading_delay.append(fresh)
        reading = reading_delay.pop(0) if len(reading_delay) > latency_ticks \
            else None
        estimator.update(reading)

        position, velocity = estimator.state
        position, velocity = predictor.predict(position, velocity, command)
        result.estimates.append(position.copy())

        command = controller(position, velocity)
        result.commands.append(command)

        for _ in range(steps_per_control):
            alpha, beta = actuator.step(*command)
            rate = ((alpha - board.tilt(data)[0]) / dt,
                    (beta - board.tilt(data)[1]) / dt)
            board.set_tilt(data, alpha, beta, rate[0], rate[1])
            mujoco.mj_step(model, data)
            estimator.predict(alpha, beta, dt)

        result.angles.append(board.tilt(data))
        result.steps = tick + 1

    return result
