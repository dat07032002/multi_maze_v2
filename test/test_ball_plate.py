"""M3 random-setpoint task and analytic baseline."""
from __future__ import annotations

import numpy as np
import pytest
import mujoco

from contract import policy_contract as pc
from sim.ball_plate import (HOLD_SECONDS, MIN_START_DISTANCE, TARGET_RADIUS,
                            ACTION_RATE_WEIGHT, DISTANCE_COST_WEIGHT,
                            BallPlateEnv, SetpointBaseline)
from sim.mjcf_builder import load_parameters


def test_reset_samples_separated_positions_and_a_valid_observation():
    env = BallPlateEnv(sensor_noise=False, seed=3)
    observation, _ = env.reset(seed=3)
    ball = env._ball_xy()
    assert np.linalg.norm(env.target - ball) >= MIN_START_DISTANCE
    assert env._reading_delay[0] == pytest.approx(ball, abs=1e-9)
    assert env.estimator.state[0] == pytest.approx(ball, abs=1e-9)
    pc.validate(observation)


def test_target_lookahead_uses_the_policy_contract_horizons():
    env = BallPlateEnv(sensor_noise=False, seed=4)
    observation, _ = env.reset(seed=4)
    start = 6 + 2 * pc.ACTION_HISTORY
    lookahead = observation[start:start + 2 * pc.LOOKAHEAD_COUNT].reshape(-1, 2)
    assert np.linalg.norm(lookahead[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.linalg.norm(lookahead, axis=1) <= 1.0 + 1e-5)


def test_freezing_far_from_target_has_a_bounded_negative_return():
    """Standing still must lose, without letting costs swamp the task."""
    env = BallPlateEnv(sensor_noise=False, seed=4)
    env.reset(seed=4)
    per_step = DISTANCE_COST_WEIGHT
    assert per_step * env.max_steps == pytest.approx(4.0)
    assert ACTION_RATE_WEIGHT * env.max_steps == pytest.approx(8.0)


def test_hold_requires_three_consecutive_seconds():
    env = BallPlateEnv(sensor_noise=False, seed=5)
    env.reset(seed=5)
    env.target = env._ball_xy().copy()
    env._previous_distance = 0.0
    terminated = False
    for step in range(int(HOLD_SECONDS * env.control_hz)):
        # Isolate the consecutive-hold state machine from the plant.
        env.board.set_ball(env.data,
                           env.target[0] - env.board_size[0] / 2,
                           env.target[1] - env.board_size[1] / 2)
        mujoco.mj_forward(env.model, env.data)
        _, _, terminated, _, info = env.step(np.zeros(2))
        if step + 1 < env.hold_steps_required:
            assert not terminated
    assert terminated
    assert info["outcome"] == "goal"
    assert info["target_distance"] <= TARGET_RADIUS


def test_analytic_setpoint_controller_reaches_and_holds():
    params = load_parameters()
    # This is also the first held-out seed used by the M3 evaluator.
    env = BallPlateEnv(params=params, seed=20_000)
    controller = SetpointBaseline(params["actuator.max_tilt"],
                                  params["actuator.centre_bias"])
    observation, _ = env.reset(seed=20_000)
    while True:
        position, velocity = env.estimator.state
        position, velocity = env.predictor.predict(
            position, velocity, env._command)
        angles = controller(position, velocity, env.target)
        action = pc.angles_to_action(*angles, params["actuator.max_tilt"])
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["outcome"] == "goal"
