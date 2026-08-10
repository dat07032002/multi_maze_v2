"""Local segment environment mechanics."""
from __future__ import annotations

import numpy as np

from contract import policy_contract as pc
from sim.segment_env import SegmentEnv, layout_for_geometry
from train.segment_dataset import generate_dataset


def _sample():
    dataset = generate_dataset(per_kind=1, per_geometry=1, seed=41)
    return dataset["geometries"][0], dataset["episodes"][0]


def test_layout_uses_geometry_as_its_route():
    geometry, _ = _sample()
    layout = layout_for_geometry(geometry)
    assert layout["waypoints"] == geometry["points_m"]
    assert layout["start_planned"] == geometry["points_m"][0]
    assert layout["goal_planned"] == geometry["points_m"][-1]


def test_reset_applies_dataset_position_velocity_and_angles():
    geometry, episode = _sample()
    env = SegmentEnv(geometry, sensor_noise=False, seed=0)
    env.reset(seed=0, options={"episode_spec": episode})
    assert np.allclose(env._ball_xy(), episode["initial_position_m"], atol=1e-6)
    assert np.allclose(env.board.ball_velocity_board(env.data)[:2],
                       episode["initial_velocity_m_s"], atol=1e-6)
    assert np.allclose(env.board.tilt(env.data),
                       episode["initial_board_angles_rad"], atol=1e-6)
    env.close()


def test_segment_can_advance_one_control_step():
    geometry, episode = _sample()
    env = SegmentEnv(geometry, sensor_noise=False, seed=0)
    observation, _ = env.reset(seed=0, options={"episode_spec": episode})
    next_observation, reward, terminated, truncated, info = env.step(
        np.zeros(2))
    assert observation.shape == next_observation.shape == (pc.OBSERVATION_SIZE,)
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    assert info["steps"] == 1
    env.close()


def test_procedural_segment_observation_uses_declared_clearance():
    geometry = {
        "id": "straight-procedural-test",
        "kind": "straight",
        "source": "procedural",
        "layout_variant": "flat_procedural",
        "points_m": [[0.05, 0.10], [0.15, 0.10]],
        "min_clearance_m": 0.004,
    }
    env = SegmentEnv(geometry, sensor_noise=False, seed=0)
    try:
        observation, _ = env.reset(seed=0)
        expected = geometry["min_clearance_m"] / pc.LOOKAHEAD_CLEARANCE_REF
        assert np.allclose(observation[-pc.LOOKAHEAD_COUNT:], expected)
        assert np.allclose(env.route.clearance, geometry["min_clearance_m"])
    finally:
        env.close()
