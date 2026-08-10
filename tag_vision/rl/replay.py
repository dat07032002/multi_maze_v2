"""Bounded, typed replay storage with atomic episode checkpoints."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, observation_size: int, action_size: int = 2,
                 *, seed: int | None = None) -> None:
        if capacity < 1 or observation_size < 1 or action_size < 1:
            raise ValueError("replay dimensions must be positive")
        self.capacity = int(capacity)
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.observations = np.empty((capacity, observation_size), np.float32)
        self.actions = np.empty((capacity, action_size), np.float32)
        self.rewards = np.empty(capacity, np.float32)
        self.next_observations = np.empty((capacity, observation_size), np.float32)
        self.terminated = np.empty(capacity, np.bool_)
        self.truncated = np.empty(capacity, np.bool_)
        self.overridden = np.empty(capacity, np.bool_)
        self.size = 0
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward: float, next_observation,
            terminated: bool, truncated: bool, overridden: bool = False) -> None:
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise ValueError("observation has wrong shape")
        if next_observation.shape != observation.shape:
            raise ValueError("next observation has wrong shape")
        if action.shape != (self.action_size,) or not np.all(np.isfinite(action)):
            raise ValueError("action has wrong shape or non-finite values")
        if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(
                next_observation)) or not np.isfinite(reward):
            raise ValueError("transition contains non-finite values")
        index = self.cursor
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)
        self.overridden[index] = bool(overridden)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not 0 < batch_size <= self.size:
            raise ValueError("batch size exceeds available replay")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {name: getattr(self, name)[indices].copy() for name in (
            "observations", "actions", "rewards", "next_observations",
            "terminated", "truncated", "overridden")}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and replace atomically so a crash cannot turn
        # the only replay checkpoint into a partial zip file.
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".",
                                         suffix=".npz", dir=target.parent)
        os.close(fd)
        try:
            np.savez_compressed(
                temporary, capacity=self.capacity,
                observation_size=self.observation_size,
                action_size=self.action_size, size=self.size,
                cursor=self.cursor,
                observations=self.observations[:self.size],
                actions=self.actions[:self.size], rewards=self.rewards[:self.size],
                next_observations=self.next_observations[:self.size],
                terminated=self.terminated[:self.size],
                truncated=self.truncated[:self.size],
                overridden=self.overridden[:self.size])
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path, *, seed: int | None = None) -> "ReplayBuffer":
        with np.load(path, allow_pickle=False) as data:
            replay = cls(int(data["capacity"]), int(data["observation_size"]),
                         int(data["action_size"]), seed=seed)
            replay.size = int(data["size"])
            replay.cursor = int(data["cursor"])
            for name in ("observations", "actions", "rewards",
                         "next_observations", "terminated", "truncated",
                         "overridden"):
                getattr(replay, name)[:replay.size] = data[name]
        return replay
