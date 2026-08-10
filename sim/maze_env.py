"""Gymnasium environment for the printed maze.

Wraps the full deployment path -- delayed noisy detection, Kalman filter,
dead-time prediction, the measured actuator, MuJoCo -- so a policy trained here
meets no new component when it reaches the rig. The same loop drives the M2
analytic baseline (see ``sim/rollout.py``), which is what makes "beats the
baseline" a claim about the policy rather than about two different plants.

Reward, in full, with an episode budget of 40:

    +30   route progress, as 30 * delta_s / L_episode
    +10   reaching the goal
     -2   falling in a hole
      0   timeout

    -0.003/step  hole proximity, 1 when the ball touches a hole edge
    -0.003/step  wall contact
    -0.004/step  cross-track error, normalised by LOCAL clearance
    -0.003/step  action rate, saturating at two 40-count commands
    -0.002/step  elapsed time, scaled by local clearance

**Time is a cost, so speed is in the objective.** Without it a policy scores the
same at 150 s as at 100 s, and no amount of tuning makes it hurry. But a FLAT
time cost teaches one global speed, and the policy carried it into tight dodges
and fell in. So the time cost is gated by local clearance: full in the open,
zero at a squeeze (TIME_COST_TIGHT..OPEN). That keeps the speed where it is safe
and removes it where it kills, which is the maze-optimal "fast in the open,
crawl at the dodge" the flat cost could not express.

Two properties of that are load-bearing rather than incidental.

**Progress is gated and monotonic.** Only increases in the furthest arc length
reached are paid, and only while the ball is within 20 mm of the route.
``route_completion`` is a *projection*, so without the gate a ball rattling
around elsewhere on the board can ratchet it up without ever travelling the
corridor. Never read progress without cross-track error beside it.

**The fall penalty is small on purpose.** Forfeited progress is the real
punishment: falling at 10 % of the route gives up 27 of progress plus the 10
bonus. An additional -10, which the plan first specified, bought nothing and
made freezing better than trying for the whole first third of the route -- the
same class of bug as the unbounded cost that once taught a policy to stand
still, just quieter.
"""
from __future__ import annotations

import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from contract import policy_contract as pc
from control.estimator import BallEstimator
from control.predictor import StatePredictor
from sim.actuator import ActuatorModel
from sim.board_state import BoardState
from sim.mjcf_builder import build_mjcf, load_layout, load_parameters
from sim.route import Route

PROGRESS_SCALE = 30.0
SUCCESS_BONUS = 10.0
FALL_PENALTY = -2.0
EPISODE_BUDGET = PROGRESS_SCALE + SUCCESS_BONUS

COST_WEIGHTS = {
    "hole_proximity": 0.003,
    "wall_contact": 0.003,
    "cross_track": 0.004,
    "action_rate": 0.003,
    # Sized against the other dense terms rather than against the budget: a
    # 3000-step episode pays 6 of the 40 on offer, and saving 30 s returns 1.2.
    # Enough to make hurrying worth something, far too little to outweigh the
    # 10 for the goal or the 30 of progress -- so ending an episode early can
    # never beat finishing it, which is what keeps a time cost from teaching a
    # policy to dive into the nearest hole.
    "time": 0.002,
}

HOLE_MARGIN = 0.008          # ball surface to hole edge; route affords 8.38 mm
PROGRESS_CORRIDOR = 0.020    # gate on crediting progress at all
GOAL_RADIUS = 0.008

# Clearance band over which the time cost ramps from off to full. Below TIGHT
# (a dodge squeeze) there is no time pressure, so the policy is free to crawl;
# above OPEN (a normal-to-open corridor) it pays the full rate and hurries.
# Falls were measured at 1.2-1.8 mm gaps, so 3 mm sits just above the fatal band.
TIME_COST_TIGHT = 0.003
TIME_COST_OPEN = 0.012


class MazeEnv(gym.Env):
    """One route-conditioned maze episode."""

    metadata = {"render_modes": []}

    def __init__(self, layout=None, params=None, control_hz: float = 20.0,
                 max_seconds: float = 60.0, sensor_noise: bool = True,
                 randomizer=None, start_fraction: float = 0.0,
                 sample_start_fraction: bool = False,
                 seed: int | None = None):
        super().__init__()
        self.base_layout = layout if layout is not None else load_layout()
        self.base_params = params if params is not None else load_parameters()
        self.control_hz = control_hz
        self.max_seconds = max_seconds
        self.max_steps = int(max_seconds * control_hz)
        self.sensor_noise = sensor_noise
        self.randomizer = randomizer
        #: 0 = start at the beginning, 0.9 = start 90 % of the way along. The
        #: reverse curriculum walks this down from near 1.
        self.start_fraction = start_fraction
        #: Draw each episode's start from ``[0, start_fraction]`` instead of
        #: pinning it there. Off by default, because a from-scratch run needs
        #: the short early stages the reverse curriculum was built to give it:
        #: over a full route the progress on offer per millimetre is small
        #: enough that falling outscores crawling, which is the dead zone
        #: ``test_the_dead_zone_is_millimetres_at_the_curriculum_start`` guards.
        #:
        #: Fine-tuning from BC has the opposite problem. The policy can already
        #: finish, so the cold start is not a risk, while a fixed start trains
        #: one slice at a time and a policy free to move forgets the rest: the
        #: v2 fine-tune held 100% window success at fraction 0.30 while
        #: full-route evaluation sat at 0%. v1 never showed it because a 1000x
        #: BC anchor left it nothing to forget.
        self.sample_start_fraction = bool(sample_start_fraction)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(pc.OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._params = dict(self.base_params)
        self._build()

    # ---- construction --------------------------------------------------
    def _build(self) -> None:
        params = self._params
        self.route = Route(self.base_layout, params)
        self.model = mujoco.MjModel.from_xml_string(
            build_mjcf(self.base_layout, params))
        self.data = mujoco.MjData(self.model)
        self.board = BoardState(self.model)
        self.dt = self.model.opt.timestep
        self.substeps = int(round(1.0 / self.control_hz / self.dt))

        self.actuator = ActuatorModel(self.dt, params=params)
        self.estimator = BallEstimator(
            measurement_std=params["camera.position_noise"],
            damping=params["ball.linear_damping"])
        self.predictor = StatePredictor(
            self.actuator, self.dt,
            sensor_latency_s=params["camera.latency"],
            damping=params["ball.linear_damping"],
            coulomb=params.get("ball.rolling_coulomb", 0.0))

        self.max_tilt = params["actuator.max_tilt"]
        self.board_size = np.array([self.base_layout["board_width"],
                                    self.base_layout["board_height"]])
        self.hole_centres = np.asarray(self.base_layout["holes"], dtype=float)
        self.hole_radii = np.asarray(self.base_layout["hole_radii"], dtype=float)
        self.ball_radius = params["ball.radius"]
        self.floor_thickness = params["maze.floor_thickness"]
        self._wall_geoms = {
            i for i in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            .startswith(("wall_", "frame_", "pad_"))
        }
        self._ball_geom = self.model.geom("ball").id

    def set_start_fraction(self, value: float) -> float:
        """Move the reverse curriculum. Called across SubprocVecEnv workers."""
        self.start_fraction = float(np.clip(value, 0.0, 0.98))
        return self.start_fraction

    def set_sample_start_fraction(self, value: bool) -> bool:
        """Toggle sampled starts. Called across SubprocVecEnv workers."""
        self.sample_start_fraction = bool(value)
        return self.sample_start_fraction

    def set_randomization_scale(self, value: float) -> float:
        """Widen the physics curriculum. Called across SubprocVecEnv workers.

        Takes effect on the next ``reset``, which is where the randomiser is
        sampled. A scale of zero disables randomisation outright rather than
        sampling a zero-width range.
        """
        if self.randomizer is None:
            return 0.0
        self.randomizer.scale = float(np.clip(value, 0.0, 1.0))
        self.randomizer.enabled = self.randomizer.scale > 0.0
        return self.randomizer.scale

    # ---- gym api -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.randomizer is not None:
            self._params = self.randomizer.sample(self.base_params, self._rng)
            self._build()

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        fraction = float(self.start_fraction)
        if self.sample_start_fraction and fraction > 0.0:
            fraction = float(self._rng.uniform(0.0, fraction))
        self._start_s = fraction * self.route.length
        start_xy = self.route.point_at(self._start_s)
        self.board.set_ball(self.data,
                            start_xy[0] - self.board_size[0] / 2,
                            start_xy[1] - self.board_size[1] / 2)
        # ``set_ball`` writes qpos. MuJoCo's derived body transforms (xpos,
        # xmat, contacts, ...) still describe the pre-reset state until a
        # forward pass.  Reading the camera before this call used to enqueue a
        # measurement from the model's default full-route start, injecting a
        # 150--190 mm jump into every non-zero curriculum reset.
        mujoco.mj_forward(self.model, self.data)
        self.actuator.reset(0.0, 0.0)
        self.estimator.reset(start_xy, (0.0, 0.0))

        self._episode_length = max(1e-6, self.route.length - self._start_s)
        # Time budget scales with how much route is actually left. A fixed 60 s
        # let an early curriculum stage -- 143 mm, about 4 s of travel for the
        # analytic baseline -- run for 24 s of wandering, which burns samples
        # and lets the dense costs accumulate past the progress on offer. The
        # 1.4x is slack over the baseline's pace, floored so the shortest stage
        # is still winnable.
        self.max_steps = int(np.clip(
            self.max_seconds * (self._episode_length / self.route.length) * 1.4,
            8.0, self.max_seconds) * self.control_hz)
        self._max_s = self._start_s
        self._history = [np.zeros(2) for _ in range(pc.ACTION_HISTORY)]
        self._last_action = np.zeros(2)
        self._command = (0.0, 0.0)
        self._steps = 0
        self._reading_delay: list = []
        self._latency_ticks = int(round(
            self._params["camera.latency"] * self.control_hz))
        self._cross_track: list[float] = []

        return self._observe(), self._info(0.0)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1, 1)
        self._command = pc.action_to_angles(action, self.max_tilt)

        wall_contact = False
        for _ in range(self.substeps):
            alpha, beta = self.actuator.step(*self._command)
            previous = self.board.tilt(self.data)
            self.board.set_tilt(self.data, alpha, beta,
                                (alpha - previous[0]) / self.dt,
                                (beta - previous[1]) / self.dt)
            mujoco.mj_step(self.model, self.data)
            self.estimator.predict(alpha, beta, self.dt)
            wall_contact = wall_contact or self._touching_wall()

        self._steps += 1
        reward, terminated, truncated, extra = self._reward(action, wall_contact)

        self._history.pop(0)
        self._history.append(action.copy())
        self._last_action = action.copy()
        return self._observe(), reward, terminated, truncated, self._info(reward, **extra)

    # ---- internals -----------------------------------------------------
    def _touching_wall(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if self._ball_geom in pair and pair & self._wall_geoms:
                return True
        return False

    def _ball_xy(self) -> np.ndarray:
        x, y, _ = self.board.ball_board(self.data)
        return np.array([x, y]) + self.board_size / 2

    def _ball_depth(self) -> float:
        return float(self.board.ball_board(self.data)[2])

    def _observe(self) -> np.ndarray:
        truth = self._ball_xy()

        fresh = None
        if self._rng.random() >= (self._params["camera.dropout_rate"]
                                  if self.sensor_noise else 0.0):
            noise = self._params["camera.position_noise"] if self.sensor_noise else 0.0
            fresh = truth + self._rng.normal(0.0, noise, size=2) if noise else truth
        self._reading_delay.append(fresh)
        reading = self._reading_delay.pop(0) \
            if len(self._reading_delay) > self._latency_ticks else None
        self.estimator.update(reading)

        position, velocity = self.estimator.state
        position, velocity = self.predictor.predict(position, velocity,
                                                    self._command)
        self._predicted = position

        s, _, _ = self.route.project(position)
        return pc.observation(position, velocity, self.board.tilt(self.data),
                              self._history,
                              self._lookahead(position, s),
                              self._lookahead_clearance(s),
                              self.board_size, self.max_tilt)

    def _lookahead(self, position: np.ndarray, s: float) -> np.ndarray:
        """Route-conditioned target points; specialised by the M3 task."""
        return self.route.lookahead(s, pc.LOOKAHEAD_SPACING,
                                    pc.LOOKAHEAD_COUNT)

    def _lookahead_clearance(self, s: float) -> np.ndarray:
        """Corridor room at each lookahead point, in metres.

        Same arc-length offsets as ``_lookahead``, so clearance is aligned to
        the path points the policy already sees. Pure route geometry, so the
        rig computes it as a table lookup from the known map -- no perception.
        """
        offsets = s + pc.LOOKAHEAD_SPACING * np.arange(1, pc.LOOKAHEAD_COUNT + 1)
        return np.array([self.route.clearance_at(min(o, self.route.length))
                         for o in offsets])

    def _reward(self, action, wall_contact):
        ball = self._ball_xy()
        s, cross, _ = self.route.project(ball)
        self._cross_track.append(abs(cross))

        # -- progress, corridor-gated and monotonic ----------------------
        progress = 0.0
        if abs(cross) <= PROGRESS_CORRIDOR and s > self._max_s:
            progress = PROGRESS_SCALE * (s - self._max_s) / self._episode_length
            self._max_s = s

        # -- bounded costs, each in [0, 1] -------------------------------
        # The flat-plate variant has no holes at all, so there is no nearest one
        # to be near: the cost is zero rather than a min over an empty array.
        if len(self.hole_centres):
            gap = float(np.min(np.linalg.norm(ball - self.hole_centres, axis=1)
                               - self.hole_radii)) - self.ball_radius
        else:
            gap = np.inf
        clearance = max(1e-4, self.route.clearance_at(s))
        costs = {
            "hole_proximity": float(np.clip(1.0 - gap / HOLE_MARGIN, 0.0, 1.0)),
            "wall_contact": 1.0 if wall_contact else 0.0,
            "cross_track": float(np.clip(abs(cross) / clearance, 0.0, 1.0)),
            "action_rate": float(np.clip(
                np.linalg.norm(action - self._last_action) / pc.ACTION_RATE_SCALE,
                0.0, 1.0)),
            # Clearance-gated: full where there is room, fading to zero at a
            # squeeze. A flat time cost taught one global speed and the policy
            # carried it into tight dodges -- measured at 113 mm/s into a 1.5 mm
            # gap, 10x what threads it -- and fell in. Removing the pressure
            # where the corridor is tight lets it crawl there while still
            # hurrying in the open, which is the actual maze-optimal behaviour.
            "time": float(np.clip(
                (clearance - TIME_COST_TIGHT)
                / (TIME_COST_OPEN - TIME_COST_TIGHT), 0.0, 1.0)),
        }
        penalty = sum(COST_WEIGHTS[k] * v for k, v in costs.items())

        reward = progress - penalty
        terminated = truncated = False
        outcome = "running"

        if self._ball_depth() < -self.floor_thickness:
            reward += FALL_PENALTY
            terminated, outcome = True, "fell"
        elif np.linalg.norm(ball - self.route.goal) < GOAL_RADIUS:
            reward += SUCCESS_BONUS
            terminated, outcome = True, "goal"
        elif self._steps >= self.max_steps:
            truncated, outcome = True, "timeout"

        return reward, terminated, truncated, {"costs": costs, "outcome": outcome}

    def _info(self, reward, costs=None, outcome="running") -> dict:
        return {
            "route_completion": (self._max_s - self._start_s) / self._episode_length,
            "arc_length": self._max_s,
            "cross_track": self._cross_track[-1] if self._cross_track else 0.0,
            "mean_cross_track": float(np.mean(self._cross_track))
            if self._cross_track else 0.0,
            "outcome": outcome,
            "costs": costs or {},
            "steps": self._steps,
        }
