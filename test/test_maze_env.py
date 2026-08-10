"""M4: the observation contract, the reward arithmetic, and the environment.

The two reward checks here are the ones that matter. Both describe failures that
are invisible in a learning curve -- the policy simply learns something else and
the curve looks fine -- and both have precedent: an unbounded cost in the
previous project taught a policy to stand still, against a budget it had
silently swamped.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from contract import policy_contract as pc
from sim import symmetry
from sim.analytic_model import ACCEL_PER_RAD
from sim.maze_env import (COST_WEIGHTS, EPISODE_BUDGET, FALL_PENALTY,
                          PROGRESS_SCALE, SUCCESS_BONUS, MazeEnv)
from sim.mjcf_builder import flat_board_layout, load_layout, load_parameters
from sim.randomization import Randomizer


@pytest.fixture(scope="module")
def env():
    return MazeEnv(seed=0)


# -- the observation contract ------------------------------------------------
def test_observation_is_27_finite_floats(env):
    observation, _ = env.reset(seed=0)
    assert observation.shape == (pc.OBSERVATION_SIZE,) == (27,)
    assert observation.dtype == np.float32
    pc.validate(observation)


@pytest.mark.parametrize("fraction", [0.92, 0.80, 0.74, 0.0])
def test_curriculum_reset_refreshes_ball_pose_before_camera(fraction):
    """A qpos write must be forwarded before the first camera sample.

    Without the forward pass every non-zero curriculum stage briefly observed
    the model's default full-route start, about 150--190 mm from the ball.
    """
    env = MazeEnv(sensor_noise=False, seed=0)
    env.set_start_fraction(fraction)
    observation, _ = env.reset(seed=0)

    expected = env.route.point_at(fraction * env.route.length)
    assert env._ball_xy() == pytest.approx(expected, abs=1e-9)
    assert env._reading_delay[0] == pytest.approx(expected, abs=1e-9)
    assert env.estimator.state[0] == pytest.approx(expected, abs=1e-9)
    pc.validate(observation)


@pytest.mark.parametrize("fraction", [0.92, 0.80, 0.74])
def test_curriculum_reset_does_not_inject_a_velocity_spike(fraction):
    env = MazeEnv(sensor_noise=False, seed=0)
    env.set_start_fraction(fraction)
    env.reset(seed=0)

    for _ in range(3):
        observation, _, _, _, _ = env.step(np.zeros(2))

    _, velocity = env.estimator.state
    assert np.linalg.norm(velocity) < 0.05
    assert np.all(np.abs(env._predicted) <= env.board_size + 0.01)
    pc.validate(observation)


def test_lookahead_is_ball_relative_and_horizon_normalised():
    """Absolute route points would make the observation say where the ball is on
    *this* board. Relative ones say what the path ahead looks like, which is
    what transfers to another maze."""
    board = np.array([0.256, 0.226])
    ball = np.array([0.10, 0.10])
    straight = np.stack([ball + [pc.LOOKAHEAD_SPACING * k, 0.0]
                         for k in range(1, pc.LOOKAHEAD_COUNT + 1)])
    clearance = np.full(pc.LOOKAHEAD_COUNT, 0.010)
    observation = pc.observation(ball, [0, 0], [0, 0],
                                 np.zeros((pc.ACTION_HISTORY, 2)),
                                 straight, clearance, board, math.radians(4.0))
    # Lookahead xy sits before the trailing clearance block, not at the end.
    start = 6 + 2 * pc.ACTION_HISTORY
    lookahead = observation[start:start + 2 * pc.LOOKAHEAD_COUNT].reshape(-1, 2)
    # A dead-straight path ahead reads as (1, 0) at every horizon.
    assert np.allclose(lookahead[:, 0], 1.0, atol=1e-5)
    assert np.allclose(lookahead[:, 1], 0.0, atol=1e-5)


def test_action_rejects_non_finite_rather_than_clipping():
    with pytest.raises(ValueError):
        pc.action_to_angles([float("nan"), 0.0], math.radians(4.0))


def test_action_maps_to_the_commanded_tilt_limit():
    alpha, beta = pc.action_to_angles([1.0, -1.0], math.radians(4.0))
    assert math.degrees(alpha) == pytest.approx(4.0)
    assert math.degrees(beta) == pytest.approx(-4.0)


# -- reward arithmetic -------------------------------------------------------
def test_dense_penalties_stay_a_minority_of_the_budget():
    """Run the analytic baseline -- a competent but imperfect agent -- and add
    up what the dense costs actually take. The plan's guard is 25 %."""
    from control.baseline import PurePursuitBaseline

    params = load_parameters()
    env = MazeEnv(params=params, seed=0)
    controller = PurePursuitBaseline(env.route, params["actuator.max_tilt"])
    observation, _ = env.reset(seed=0)

    spent = 0.0
    while True:
        position, velocity = env.estimator.state
        position, velocity = env.predictor.predict(position, velocity,
                                                   env._command)
        action = pc.angles_to_action(*controller(position, velocity),
                                     params["actuator.max_tilt"])
        observation, _, terminated, truncated, info = env.step(action)
        spent += sum(COST_WEIGHTS[k] * v for k, v in info["costs"].items())
        if terminated or truncated:
            break

    assert spent < 0.25 * EPISODE_BUDGET, (
        f"dense penalties took {spent:.2f} of a {EPISODE_BUDGET:.0f} budget "
        f"({spent / EPISODE_BUDGET:.0%})")


def test_freezing_is_only_better_than_falling_very_early():
    """The dead zone is real, and small. The plan overstated this.

    An agent that falls scores ``PROGRESS_SCALE * fraction + FALL_PENALTY``
    against 0 for never moving, so freezing wins below
    ``-FALL_PENALTY / PROGRESS_SCALE``. At the -10 the plan first specified that
    was 33 % of the route -- freezing was optimal across the whole first third.
    At -2 it is 6.7 %, and the claim that it vanished entirely was wrong.

    What makes 6.7 % harmless is that it is 6.7 % of the *episode's* route, and
    the reverse curriculum starts with almost none of the route left to run.
    """
    break_even = -FALL_PENALTY / PROGRESS_SCALE
    assert break_even == pytest.approx(2 / 30)
    assert break_even < 0.10

    for fraction in (0.10, 0.3, 0.5, 0.9):
        assert PROGRESS_SCALE * fraction + FALL_PENALTY > 0.0


def test_the_dead_zone_is_millimetres_at_the_curriculum_start():
    """Which is what makes the reverse curriculum load-bearing rather than a
    convenience: it keeps available progress well above the break-even."""
    env = MazeEnv(seed=0)
    env.set_start_fraction(0.92)
    env.reset(seed=0)
    dead_zone = (-FALL_PENALTY / PROGRESS_SCALE) * env._episode_length
    assert dead_zone < 0.010, (
        f"dead zone is {dead_zone * 1000:.1f} mm at the first curriculum stage")


def test_reaching_the_goal_is_worth_the_whole_budget():
    assert PROGRESS_SCALE + SUCCESS_BONUS == EPISODE_BUDGET


def test_progress_is_gated_on_the_corridor(env):
    """Completion is a projection onto the route: without the gate a ball
    rattling around elsewhere ratchets it up without travelling the corridor."""
    observation, _ = env.reset(seed=0)
    start = env._max_s

    # Somewhere genuinely off-route. The board centre is not it -- the route
    # runs close enough to the middle that dropping the ball there credited
    # 449 mm of progress, which is exactly the ratcheting this gate prevents.
    grid = np.stack(np.meshgrid(
        np.linspace(0.02, env.board_size[0] - 0.02, 40),
        np.linspace(0.02, env.board_size[1] - 0.02, 40)), axis=-1).reshape(-1, 2)
    distances = np.linalg.norm(
        grid[:, None, :] - env.route.points[None, :, :], axis=2).min(axis=1)
    far = grid[int(np.argmax(distances))]
    assert distances.max() > 0.025, "no point on this board is far from the route"

    env.board.set_ball(env.data, far[0] - env.board_size[0] / 2,
                       far[1] - env.board_size[1] / 2)
    env.step(np.zeros(2))
    assert env._max_s == pytest.approx(start, abs=1e-6)


def test_action_rate_cost_is_written_in_hardware_units():
    """Saturates at two 40-count commands, not at an arbitrary number."""
    assert pc.ACTION_QUANTUM == pytest.approx(0.19 / 4.0)
    assert pc.ACTION_RATE_SCALE == pytest.approx(2.0 * pc.ACTION_QUANTUM)


# -- environment mechanics ---------------------------------------------------
def test_episode_terminates_on_a_hole():
    env = MazeEnv(seed=0)
    env.reset(seed=0)
    layout = load_layout()
    hx, hy = layout["holes"][0]
    env.board.set_ball(env.data, hx - env.board_size[0] / 2,
                       hy - env.board_size[1] / 2)
    for _ in range(40):
        _, reward, terminated, _, info = env.step(np.zeros(2))
        if terminated:
            break
    assert terminated and info["outcome"] == "fell"


def test_reverse_curriculum_shortens_the_episode(env):
    env.set_start_fraction(0.9)
    env.reset(seed=0)
    near = env._episode_length
    env.set_start_fraction(0.0)
    env.reset(seed=0)
    assert env._episode_length > near * 5
    env.set_start_fraction(0.0)


def test_flat_variant_has_no_holes_to_be_near():
    """The maze env is reused for the flat-plate task, where a min over an
    empty array of holes is not a small edge case but a crash."""
    env = MazeEnv(layout=flat_board_layout(), seed=0)
    env.reset(seed=0)
    for _ in range(20):
        _, _, _, _, info = env.step(np.zeros(2))
    assert info["costs"]["hole_proximity"] == 0.0


def test_randomizer_only_touches_declared_parameters():
    params = load_parameters()
    sampled = Randomizer().sample(params, np.random.default_rng(0))
    changed = {k for k in params if sampled[k] != params[k]}
    allowed = set(Randomizer().spec) | {"sim.wall_dampratio"}
    assert changed <= allowed, f"randomiser altered {changed - allowed}"


def test_randomizer_scale_zero_is_the_nominal_model():
    params = load_parameters()
    sampled = Randomizer(scale=0.0).sample(params, np.random.default_rng(0))
    assert sampled == params


def test_default_randomizer_uses_contracted_training_range():
    params = load_parameters()
    randomizer = Randomizer()
    assert randomizer.scale == 0.10
    nominal, bounds = randomizer.spec["ball.rolling_friction_length"]
    expected_low = nominal + (bounds[0] - nominal) * randomizer.scale
    expected_high = nominal + (bounds[1] - nominal) * randomizer.scale
    samples = [randomizer.sample(params, np.random.default_rng(seed))[
        "ball.rolling_friction_length"] for seed in range(100)]
    assert min(samples) >= expected_low
    assert max(samples) <= expected_high


def test_full_stress_randomizer_remains_available():
    params = load_parameters()
    randomizer = Randomizer(scale=1.0)
    nominal, bounds = randomizer.spec["camera.latency"]
    samples = [randomizer.sample(params, np.random.default_rng(seed))[
        "camera.latency"] for seed in range(100)]
    assert min(samples) >= bounds[0]
    assert max(samples) <= bounds[1]


# -- symmetry ----------------------------------------------------------------
def test_mirror_is_an_exact_symmetry_of_the_plate_dynamics():
    """Reflecting the state and the action reflects the acceleration exactly.

    This is what makes the augmentation valid on the flat plate. It says nothing
    about the maze, whose walls and holes are not mirror-symmetric -- see the
    module docstring for why that matters to the reward.
    """
    for alpha_deg, beta_deg in ((2.0, 3.0), (-1.0, 4.0), (0.5, -2.5)):
        alpha, beta = math.radians(alpha_deg), math.radians(beta_deg)
        ax = ACCEL_PER_RAD * math.sin(beta)
        ay = -ACCEL_PER_RAD * math.sin(alpha)
        # Mirroring in x flips beta, which flips the x acceleration only.
        assert ACCEL_PER_RAD * math.sin(-beta) == pytest.approx(-ax)
        assert -ACCEL_PER_RAD * math.sin(alpha) == pytest.approx(ay)


def test_mirroring_an_observation_is_an_involution():
    rng = np.random.default_rng(0)
    observation = rng.normal(size=pc.OBSERVATION_SIZE).astype(np.float32)
    action = rng.uniform(-1, 1, size=2).astype(np.float32)
    for mirror in (symmetry.mirror_x, symmetry.mirror_y):
        once = mirror(observation, action)
        twice = mirror(*once)
        assert np.allclose(twice[0], observation, atol=1e-6)
        assert np.allclose(twice[1], action, atol=1e-6)


def test_augment_returns_four_variants():
    rng = np.random.default_rng(0)
    observation = rng.normal(size=pc.OBSERVATION_SIZE).astype(np.float32)
    action = rng.uniform(-1, 1, size=2).astype(np.float32)
    variants = symmetry.augment(observation, action)
    assert len(variants) == 4
    assert np.allclose(variants[0][0], observation)


def test_mirroring_a_layout_preserves_its_geometry():
    """The valid way to use the symmetry on the maze: a real second board,
    not a relabelling of this one."""
    layout = load_layout()
    mirrored = symmetry.mirror_layout(layout, axis=0)
    W = layout["board_width"]
    assert len(mirrored["holes"]) == len(layout["holes"])
    for original, flipped in zip(layout["holes"], mirrored["holes"]):
        assert flipped[0] == pytest.approx(W - original[0])
        assert flipped[1] == pytest.approx(original[1])
    for lo, hi, _ in mirrored["walls_h"]:
        assert hi >= lo, "mirroring left a wall run reversed"


def test_randomization_scale_setter_moves_the_randomizer():
    from sim.randomization import Randomizer
    env = MazeEnv(randomizer=Randomizer(scale=0.10), seed=0)
    try:
        assert env.set_randomization_scale(0.25) == pytest.approx(0.25)
        assert env.randomizer.scale == pytest.approx(0.25)
        assert env.randomizer.enabled
        # Zero disables rather than sampling a zero-width range.
        assert env.set_randomization_scale(0.0) == pytest.approx(0.0)
        assert not env.randomizer.enabled
        # Clamped to the documented full range.
        assert env.set_randomization_scale(5.0) == pytest.approx(1.0)
    finally:
        env.close()


def test_randomization_scale_setter_is_safe_without_a_randomizer():
    env = MazeEnv(randomizer=None, seed=0)
    try:
        assert env.set_randomization_scale(0.25) == 0.0
    finally:
        env.close()


def test_sampled_starts_cover_the_whole_route_below_the_curriculum_stage():
    """A fixed start trains one slice; sampling keeps the rest in distribution.

    The v2 fine-tune held 100% window success at start fraction 0.30 and scored
    0% on full-route evaluation, having forgotten the first 30% it no longer
    visited. Off by default so a from-scratch run keeps its short early stages.
    """
    env = MazeEnv(max_seconds=20.0, start_fraction=0.8,
                  sample_start_fraction=True, seed=0)
    try:
        starts = []
        for seed in range(40):
            env.reset(seed=seed)
            starts.append(env._start_s / env.route.length)
        assert min(starts) < 0.2, "sampling never reaches the route start"
        assert max(starts) > 0.6, "sampling never reaches the curriculum stage"
        assert max(starts) <= 0.8 + 1e-9, "sampling ran past the stage"
    finally:
        env.close()


def test_sampling_is_off_by_default_and_pins_the_start():
    env = MazeEnv(max_seconds=20.0, start_fraction=0.8, seed=0)
    try:
        for seed in range(5):
            env.reset(seed=seed)
            assert env._start_s == pytest.approx(0.8 * env.route.length)
    finally:
        env.close()


def test_evaluation_start_is_unaffected_by_sampling():
    # start_fraction 0 is every evaluation path: [0, 0] is exactly 0 either way.
    for sampling in (False, True):
        env = MazeEnv(max_seconds=20.0, start_fraction=0.0,
                      sample_start_fraction=sampling, seed=0)
        try:
            for seed in range(3):
                env.reset(seed=seed)
                assert env._start_s == pytest.approx(0.0)
        finally:
            env.close()


def test_time_cost_is_clearance_gated():
    """Speed must be in the objective, but only where the corridor has room.

    A flat time cost taught one global speed that killed the policy at tight
    dodges; gating by clearance removes the pressure exactly there.
    """
    from sim.maze_env import TIME_COST_TIGHT, TIME_COST_OPEN
    assert COST_WEIGHTS["time"] > 0.0
    assert 0.0 < TIME_COST_TIGHT < TIME_COST_OPEN
    env = MazeEnv(max_seconds=20.0, seed=0)
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        # The gated factor lives in [0, 1]; open corridors approach 1, a
        # squeeze approaches 0.
        assert 0.0 <= info["costs"]["time"] <= 1.0
        # Still small enough it can never beat finishing, even at full rate.
        assert COST_WEIGHTS["time"] * 3000 < SUCCESS_BONUS
    finally:
        env.close()


def test_time_cost_vanishes_at_a_squeeze_and_is_full_in_the_open():
    """Directly pin the gate the fall-fix depends on."""
    from sim.maze_env import TIME_COST_TIGHT, TIME_COST_OPEN
    env = MazeEnv(max_seconds=20.0, seed=0)
    try:
        env.reset(seed=0)
        # Monkeypatch the projected clearance the reward reads.
        wide = lambda s: TIME_COST_OPEN + 0.05
        tight = lambda s: TIME_COST_TIGHT - 0.001
        env.route.clearance_at = wide
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["costs"]["time"] == pytest.approx(1.0)
        env.route.clearance_at = tight
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["costs"]["time"] == pytest.approx(0.0)
    finally:
        env.close()
