"""CEM-MPC teacher smoke and safety checks."""
from __future__ import annotations

import numpy as np

from control.mpc_teacher import CEMConfig, CEMMPCTeacher
from sim.segment_env import SegmentEnv
from train.segment_dataset import generate_dataset


def _env_and_episode():
    dataset = generate_dataset(per_kind=1, per_geometry=1, seed=51)
    geometry = dataset["geometries"][0]
    episode = dataset["episodes"][0]
    env = SegmentEnv(geometry, sensor_noise=False, seed=0)
    observation, _ = env.reset(seed=0, options={"episode_spec": episode})
    return env, observation


def test_teacher_action_is_finite_and_bounded():
    env, observation = _env_and_episode()
    teacher = CEMMPCTeacher(env, seed=0, config=CEMConfig(
        horizon_steps=8, candidates=24, iterations=2, elites=4))
    action = teacher(observation)
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)
    env.close()


def test_teacher_is_deterministic_for_a_fixed_seed():
    env, observation = _env_and_episode()
    config = CEMConfig(horizon_steps=8, candidates=24,
                       iterations=2, elites=4)
    first = CEMMPCTeacher(env, seed=7, config=config)(observation)
    second = CEMMPCTeacher(env, seed=7, config=config)(observation)
    assert np.allclose(first, second)
    env.close()
