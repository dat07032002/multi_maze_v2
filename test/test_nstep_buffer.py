import gymnasium as gym
import numpy as np
import pytest

from train.nstep_buffer import NStepReplayBuffer

OBS = gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
ACT = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
GAMMA = 0.9


def make_buffer(size=32, n_steps=3, n_envs=1):
    return NStepReplayBuffer(size, OBS, ACT, n_envs=n_envs,
                             n_steps=n_steps, gamma=GAMMA,
                             handle_timeout_termination=True)


def add(buffer, index, reward, done=False, timeout=False, n_envs=1):
    """One transition whose observation encodes its own index."""
    obs = np.full((n_envs, 2), index, dtype=np.float32)
    nxt = np.full((n_envs, 2), index + 1, dtype=np.float32)
    buffer.add(obs, nxt, np.zeros((n_envs, 1), dtype=np.float32),
               np.full(n_envs, reward, dtype=np.float32),
               np.full(n_envs, done, dtype=bool),
               [{"TimeLimit.truncated": timeout}] * n_envs)


def test_walk_accumulates_discounted_rewards_and_gamma_to_the_n():
    buffer = make_buffer(n_steps=3)
    for index in range(10):
        add(buffer, index, reward=1.0)

    sample = buffer._get_samples(np.array([2]))
    expected = 1.0 + GAMMA + GAMMA ** 2
    assert sample.rewards.item() == pytest.approx(expected, rel=1e-6)
    assert sample.discounts.item() == pytest.approx(GAMMA ** 3, rel=1e-6)
    # Bootstrap hangs off the state three steps on, not one.
    assert sample.next_observations[0, 0].item() == pytest.approx(5.0)


def test_one_step_matches_plain_replay_semantics():
    buffer = make_buffer(n_steps=1)
    for index in range(6):
        add(buffer, index, reward=2.0)

    sample = buffer._get_samples(np.array([3]))
    assert sample.rewards.item() == pytest.approx(2.0)
    assert sample.discounts.item() == pytest.approx(GAMMA)
    assert sample.next_observations[0, 0].item() == pytest.approx(4.0)


def test_walk_stops_at_a_terminal_and_reports_it():
    buffer = make_buffer(n_steps=4)
    add(buffer, 0, reward=1.0)
    add(buffer, 1, reward=1.0, done=True)          # fell: real terminal
    for index in range(2, 8):
        add(buffer, index, reward=1.0)

    sample = buffer._get_samples(np.array([0]))
    # Two rewards only; the walk must not cross into the next episode.
    assert sample.rewards.item() == pytest.approx(1.0 + GAMMA, rel=1e-6)
    assert sample.discounts.item() == pytest.approx(GAMMA ** 2, rel=1e-6)
    assert sample.dones.item() == pytest.approx(1.0)


def test_timeout_stops_the_walk_but_still_bootstraps():
    buffer = make_buffer(n_steps=4)
    add(buffer, 0, reward=1.0)
    add(buffer, 1, reward=1.0, done=True, timeout=True)
    for index in range(2, 8):
        add(buffer, index, reward=1.0)

    sample = buffer._get_samples(np.array([0]))
    assert sample.rewards.item() == pytest.approx(1.0 + GAMMA, rel=1e-6)
    # A time limit is not a real terminal: the value beyond it still counts.
    assert sample.dones.item() == pytest.approx(0.0)


def test_walk_never_crosses_the_write_head():
    buffer = make_buffer(size=8, n_steps=5)
    for index in range(6):
        add(buffer, index, reward=1.0)

    # pos is 6; a walk from 4 has two valid steps, not five.
    sample = buffer._get_samples(np.array([4]))
    assert sample.rewards.item() == pytest.approx(1.0 + GAMMA, rel=1e-6)
    assert sample.discounts.item() == pytest.approx(GAMMA ** 2, rel=1e-6)


def test_walk_respects_the_write_head_after_wraparound():
    buffer = make_buffer(size=8, n_steps=5)
    for index in range(11):                        # wraps: pos lands at 3
        add(buffer, index, reward=1.0)
    assert buffer.full and buffer.pos == 3

    # Index 1 is two steps from the head, and the walk must wrap 7 -> 0.
    sample = buffer._get_samples(np.array([1]))
    assert sample.rewards.item() == pytest.approx(1.0 + GAMMA, rel=1e-6)
    # From 6 the walk wraps 7 -> 0 -> 1 -> 2 and stops on the slot before the
    # head, so all five steps are available.
    sample = buffer._get_samples(np.array([6]))
    assert sample.rewards.item() == pytest.approx(
        sum(GAMMA ** k for k in range(5)), rel=1e-6)
    assert sample.next_observations[0, 0].item() == pytest.approx(11.0)


def test_rejects_configurations_it_cannot_walk():
    with pytest.raises(ValueError, match="n_steps"):
        make_buffer(n_steps=0)
    with pytest.raises(ValueError, match="handle_timeout_termination"):
        NStepReplayBuffer(32, OBS, ACT, n_envs=1, n_steps=3, gamma=GAMMA,
                          handle_timeout_termination=False)
