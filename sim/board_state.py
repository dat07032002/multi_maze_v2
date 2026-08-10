"""Reading and driving the simulated board.

Small on purpose, but two things in here are easy to get wrong and expensive to
get wrong quietly.

**Driving the hinges sets qvel as well as qpos.** Writing ``qpos`` alone and
leaving ``qvel`` at zero makes a moving plate look stationary to the contact
solver: the ball stops feeling the drag a tilting plate exerts on it, and each
control step lands as an impulse instead of a motion. The board is kinematic
here -- the measured actuator dynamics live in ``sim/actuator.py`` (M1), and
this module only applies whatever angle that produces.

**Ball position is reported in the board frame**, which is what the vision stack
measures and what the policy sees. MuJoCo tracks the ball in world coordinates,
and with the board tilted the two differ by a rotation; comparing a world
coordinate against a layout coordinate is silently wrong by up to a few
millimetres at 4 degrees.
"""
from __future__ import annotations

import mujoco
import numpy as np


class BoardState:
    """Indices and frame conversions for one compiled maze model."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.tilt_x = model.joint("tilt_x").id
        self.tilt_y = model.joint("tilt_y").id
        self.qpos_x = model.jnt_qposadr[self.tilt_x]
        self.qpos_y = model.jnt_qposadr[self.tilt_y]
        self.dof_x = model.jnt_dofadr[self.tilt_x]
        self.dof_y = model.jnt_dofadr[self.tilt_y]
        self.ball_body = model.body("ball").id
        self.ball_qpos = model.jnt_qposadr[model.joint("ball_free").id]
        self.ball_dof = model.jnt_dofadr[model.joint("ball_free").id]
        self.board_body = model.body("board").id

    # ---- driving -------------------------------------------------------
    def set_tilt(self, data: mujoco.MjData, alpha: float, beta: float,
                 alpha_rate: float = 0.0, beta_rate: float = 0.0) -> None:
        """Place the board at ``(alpha, beta)`` with the given angular rates."""
        data.qpos[self.qpos_x] = alpha
        data.qpos[self.qpos_y] = beta
        data.qvel[self.dof_x] = alpha_rate
        data.qvel[self.dof_y] = beta_rate
        data.ctrl[:] = (alpha, beta)

    def tilt(self, data: mujoco.MjData) -> tuple[float, float]:
        return float(data.qpos[self.qpos_x]), float(data.qpos[self.qpos_y])

    # ---- reading -------------------------------------------------------
    def board_rotation(self, data: mujoco.MjData) -> np.ndarray:
        """World-from-board rotation, i.e. ``Rx(alpha) @ Ry(beta)``."""
        return np.array(data.xmat[self.board_body]).reshape(3, 3)

    def ball_world(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(data.xpos[self.ball_body])

    def ball_board(self, data: mujoco.MjData) -> np.ndarray:
        """Ball centre in the board frame, origin at the board centre."""
        return self.board_rotation(data).T @ self.ball_world(data)

    def ball_velocity_board(self, data: mujoco.MjData) -> np.ndarray:
        world = np.array(data.qvel[self.ball_dof:self.ball_dof + 3])
        return self.board_rotation(data).T @ world

    def set_ball(self, data: mujoco.MjData, x: float, y: float,
                 z: float | None = None) -> None:
        """Place the ball at board-frame ``(x, y)``, at rest."""
        radius = float(self.model.geom("ball").size[0])
        local = np.array([x, y, radius if z is None else z])
        data.qpos[self.ball_qpos:self.ball_qpos + 3] = (
            self.board_rotation(data) @ local)
        data.qpos[self.ball_qpos + 3:self.ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[self.ball_dof:self.ball_dof + 6] = 0.0

    def set_ball_velocity_board(self, data: mujoco.MjData, vx: float,
                                vy: float, vz: float = 0.0) -> None:
        """Set a no-slip ball velocity expressed in the board frame."""
        linear_local = np.array([vx, vy, vz], dtype=float)
        radius = float(self.model.geom("ball").size[0])
        # v = omega x (0, 0, -r), hence omega=(vy/r, -vx/r, 0).
        angular_local = np.array([vy / radius, -vx / radius, 0.0])
        rotation = self.board_rotation(data)
        data.qvel[self.ball_dof:self.ball_dof + 3] = rotation @ linear_local
        data.qvel[self.ball_dof + 3:self.ball_dof + 6] = rotation @ angular_local

    # ---- diagnostics ---------------------------------------------------
    def max_penetration(self, data: mujoco.MjData) -> float:
        """Deepest current contact interpenetration, in metres (>= 0)."""
        if data.ncon == 0:
            return 0.0
        return float(max(0.0, -min(data.contact[i].dist
                                   for i in range(data.ncon))))


class TiltDriver:
    """Slews the board toward a target at the measured maximum rate.

    A stand-in for the full actuator model (M1), carrying only the piece that
    cannot be left out even in a geometry test: the rate limit. Commanding a
    tilt as an instantaneous jump in ``qpos`` teleports the plate through up to
    8 degrees in one millisecond, which slams the floor into the ball -- on this
    model that launches it about 10 mm into the air with no contacts at all, and
    then it lands anywhere. The rig physically cannot do that; it manages
    6.9 deg/s on roll and 8.2 on pitch.

    Rates are passed through to ``set_tilt`` so the contact solver sees a plate
    that is moving, not one that has already arrived.
    """

    def __init__(self, state: BoardState, timestep: float,
                 max_rate: tuple[float, float] = (0.12063788738196578,
                                                  0.1430373149468565)):
        self.state = state
        self.dt = timestep
        self.max_rate = max_rate
        self.alpha = 0.0
        self.beta = 0.0

    def reset(self, alpha: float = 0.0, beta: float = 0.0) -> None:
        self.alpha, self.beta = alpha, beta

    def step(self, data: mujoco.MjData, target_alpha: float,
             target_beta: float) -> None:
        rates = []
        for axis, (current, target) in enumerate(
                ((self.alpha, target_alpha), (self.beta, target_beta))):
            limit = self.max_rate[axis] * self.dt
            delta = max(-limit, min(limit, target - current))
            rates.append(delta / self.dt)
            if axis == 0:
                self.alpha = current + delta
            else:
                self.beta = current + delta
        self.state.set_tilt(data, self.alpha, self.beta, rates[0], rates[1])


def measure_restitution(model: mujoco.MjModel, data: mujoco.MjData,
                        state: BoardState, speed: float = 0.30) -> float:
    """Coefficient of restitution the model actually delivers against a wall.

    MuJoCo takes a solref damping ratio, not a restitution coefficient, so
    ``ball.wall_restitution`` is a target rather than a setting and the only
    honest way to know the realised value is to bounce a ball and look. Speeds
    matter: this is measured at 0.30 m/s, in the middle of the 0-0.50 m/s range
    the board can produce at the +-4 degree command limit.

    Separation speed is taken after contact **ends**, not at the first frame the
    velocity turns around. A soft contact reverses velocity gradually, so
    sampling mid-bounce reads far too low -- which is what made an earlier
    sweep report 0.007 for a contact that is visibly springy.
    """
    for _ in range(int(round(0.3 / model.opt.timestep))):
        state.set_tilt(data, 0.0, 0.0)
        mujoco.mj_step(model, data)
    data.qvel[state.ball_dof:state.ball_dof + 6] = 0.0
    data.qvel[state.ball_dof] = speed

    was_touching, impact = False, None
    for _ in range(int(round(4.0 / model.opt.timestep))):
        state.set_tilt(data, 0.0, 0.0)
        mujoco.mj_step(model, data)
        touching = data.ncon > 1  # one contact is always the floor
        vx = state.ball_velocity_board(data)[0]
        if touching and not was_touching:
            impact = abs(vx)
        elif was_touching and not touching and impact:
            return abs(vx) / impact
        was_touching = touching
    return 0.0


def settle(model: mujoco.MjModel, data: mujoco.MjData, state: BoardState,
           seconds: float = 0.3, alpha: float = 0.0, beta: float = 0.0) -> None:
    """Hold a tilt until transients die, then zero the ball's velocity."""
    for _ in range(int(round(seconds / model.opt.timestep))):
        state.set_tilt(data, alpha, beta)
        mujoco.mj_step(model, data)
    data.qvel[state.ball_dof:state.ball_dof + 6] = 0.0
    mujoco.mj_forward(model, data)
