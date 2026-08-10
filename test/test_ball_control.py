import numpy as np

from tag_vision.control.ball_dynamics import (
    BallDynamicsModel, fit_ball_dynamics)
from tag_vision.control.ball_state import BallState, BallStateFilter
from tag_vision.control.cem_mpc import RouteMPC
from tag_vision.control.reset_brake import AdaptiveResetBrake, ResetPhase
from tag_vision.control.residual_policy import ResidualActionMixer
from tag_vision.control.fixed_reset_brake import FixedResetBrake, FixedResetPhase


def model():
    return BallDynamicsModel(
        np.array([[80.0, 5.0], [-3.0, 75.0]]),
        np.array([[0.5, 0.0], [0.0, 0.4]]), np.array([1.0, -2.0]))


def test_ball_filter_tracks_and_resets_after_gap():
    filt = BallStateFilter(position_gain=1.0, velocity_gain=0.5,
                           reset_gap_s=0.3)
    assert filt.update(0.0) is None
    first = filt.update(0.1, [10, 20])
    assert np.allclose(first.velocity_mm_s, 0)
    moving = filt.update(0.2, [13, 20])
    assert moving.velocity_mm_s[0] > 0
    predicted = filt.update(0.25)
    assert not predicted.observed and predicted.measurement_age_s > 0
    reset = filt.update(0.6, [50, 40])
    assert np.allclose(reset.position_mm, [50, 40])
    assert np.allclose(reset.velocity_mm_s, 0)


def test_ball_filter_suppresses_stationary_pixel_jitter():
    filt = BallStateFilter(measurement_noise_mm=0.35,
                           stationary_speed_mm_s=2.0)
    filt.update(0.0, [100.0, 100.0])
    for index, offset in enumerate((0.2, -0.2, 0.3, -0.1), start=1):
        state = filt.update(index * 0.05, [100.0 + offset, 100.0])
    assert state.speed_mm_s == 0.0


def test_fit_recovers_cross_axis_model():
    rng = np.random.default_rng(4)
    truth = model()
    velocity = rng.normal(0, 30, (300, 2))
    tilt = rng.uniform(-2, 2, (300, 2))
    acceleration = np.array([
        truth.acceleration(v, u) for v, u in zip(velocity, tilt)])
    fitted = fit_ball_dynamics(velocity, tilt, acceleration)
    assert np.allclose(fitted.acceleration_per_tilt,
                       truth.acceleration_per_tilt, atol=1e-5)
    assert np.allclose(fitted.velocity_damping,
                       truth.velocity_damping, atol=1e-5)


def test_reset_brake_opposes_velocity_and_waits_for_hold():
    brake = AdaptiveResetBrake(model(), settle_hold_s=0.5)
    state = BallState(0.0, np.array([100., 100.]), np.array([100., 0.]),
                      0.0, True)
    command = brake.update(state)
    assert command.phase == ResetPhase.BRAKING
    acceleration = model().acceleration(state.velocity_mm_s, command.tilt_deg)
    assert np.dot(acceleration, state.velocity_mm_s) < 0
    slow = BallState(1.0, state.position_mm, np.zeros(2), 0.0, True)
    assert brake.update(slow).phase == ResetPhase.SETTLING
    ready = BallState(1.6, state.position_mm, np.zeros(2), 0.0, True)
    assert brake.update(ready).episode_ready
    stale = BallState(1.7, state.position_mm, np.zeros(2), 0.5, False)
    assert brake.update(stale).phase == ResetPhase.LOST
    assert np.allclose(brake.update(stale).tilt_deg, 0)


def test_reset_brake_times_out_to_level():
    brake = AdaptiveResetBrake(model(), max_brake_duration_s=0.2)
    moving = lambda t: BallState(  # noqa: E731
        t, np.array([100., 100.]), np.array([100., 0.]), 0.0, True)
    assert brake.update(moving(0.0)).phase == ResetPhase.BRAKING
    timed_out = brake.update(moving(0.3))
    assert timed_out.phase == ResetPhase.TIMEOUT
    assert np.allclose(timed_out.tilt_deg, 0)


def test_residual_starts_with_zero_authority():
    mixer = ResidualActionMixer(initial_authority=0.0)
    assert np.allclose(mixer.mix([1.0, -1.0], [1.0, 1.0]), [1.0, -1.0])
    mixer.set_authority(1.0)
    assert np.allclose(mixer.mix([1.0, -1.0], [1.0, 1.0]),
                       [1.35, -0.65])


def test_fixed_reset_brake_arms_brakes_and_settles():
    brake = FixedResetBrake([-0.5, 1.0], arm_after_absent_s=0.2,
                            settle_hold_s=0.3)
    assert brake.update(None, 0.0).phase == FixedResetPhase.WAITING_FOR_DROP
    armed = brake.update(None, 0.25)
    assert armed.phase == FixedResetPhase.ARMED
    assert np.allclose(armed.tilt_deg, [-0.5, 1.0])
    fast = BallState(0.3, np.array([100., 200.]), np.array([-60., -20.]),
                     0.0, True)
    command = brake.update(fast, 0.3)
    assert command.phase == FixedResetPhase.BRAKING
    assert np.allclose(command.tilt_deg, [-0.5, 1.0])
    slow = BallState(0.7, fast.position_mm, np.zeros(2), 0.0, True)
    assert brake.update(slow, 0.7).phase == FixedResetPhase.SETTLING
    assert brake.update(slow, 1.1).episode_ready


def test_fixed_reset_brake_triggers_immediately_on_reappearance():
    brake = FixedResetBrake([-0.5, 1.0], arm_after_absent_s=0.2)
    brake.update(None, 0.0)
    brake.update(None, 0.25)
    first_detection = BallState(
        0.3, np.array([100., 200.]), np.zeros(2), 0.0, True)
    command = brake.update(first_detection, 0.3)
    assert command.phase == FixedResetPhase.BRAKING
    assert np.allclose(command.tilt_deg, [-0.5, 1.0])


def test_fixed_reset_brake_recovers_from_timeout_when_ball_stops():
    brake = FixedResetBrake(
        [-0.5, 1.0], arm_after_absent_s=0.2,
        minimum_brake_duration_s=0.1, max_brake_duration_s=0.4,
        settle_hold_s=0.3)
    brake.update(None, 0.0)
    brake.update(None, 0.25)
    moving = BallState(
        0.3, np.array([100., 200.]), np.array([80., 0.]), 0.0, True)
    assert brake.update(moving, 0.3).phase == FixedResetPhase.BRAKING
    assert brake.update(moving, 0.8).phase == FixedResetPhase.TIMEOUT
    stopped = BallState(
        0.9, moving.position_mm, np.zeros(2), 0.0, True)
    settling = brake.update(stopped, 0.9)
    assert settling.phase == FixedResetPhase.SETTLING
    assert np.allclose(settling.tilt_deg, 0.0)
    assert brake.update(stopped, 1.25).episode_ready


def test_fixed_reset_brake_rearms_after_timeout_and_second_loss():
    brake = FixedResetBrake(
        [-0.5, 1.0], arm_after_absent_s=0.2,
        minimum_brake_duration_s=0.1, max_brake_duration_s=0.4)
    brake.update(None, 0.0)
    brake.update(None, 0.25)
    moving = BallState(
        0.3, np.array([100., 200.]), np.array([80., 0.]), 0.0, True)
    brake.update(moving, 0.3)
    assert brake.update(moving, 0.8).phase == FixedResetPhase.TIMEOUT
    brake.update(None, 0.9)
    rearmed = brake.update(None, 1.15)
    assert rearmed.phase == FixedResetPhase.ARMED
    assert np.allclose(rearmed.tilt_deg, [-0.5, 1.0])


def test_mpc_moves_toward_route_without_collision():
    simple = BallDynamicsModel(np.eye(2) * 100.0, np.eye(2) * 0.4,
                               np.zeros(2))
    occupied = np.zeros((100, 100), dtype=bool)
    route = np.column_stack((np.linspace(10, 80, 30), np.full(30, 50.0)))
    mpc = RouteMPC(simple, route, occupied, board_height_mm=100,
                   horizon=6, candidates=128, iterations=3, seed=3)
    result = mpc.command([10, 50, 0, 0])
    assert result.tilt_deg[0] > 0
    assert abs(result.tilt_deg[1]) < 1.5
    assert result.predicted_states[-1, 0] > 10
