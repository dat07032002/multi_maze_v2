"""N-step replay for the route-conditioned fine-tune.

``gamma=0.995`` at 20 Hz gives the critic an effective horizon of
``1/(1-gamma) = 200`` steps, ten seconds, against episodes that run 150. The
goal bonus arriving 3000 steps out is worth ``10 * 0.995**3000 ~ 3e-6`` at
episode start, so a one-step critic cannot see the goal at all, and reward
information crawls backwards one step per gradient update.

N-step returns cut both: the sample carries ``n`` real rewards instead of one,
and its bootstrap discount is ``gamma**n``. ``BehaviorRegularizedSAC.train``
already reads ``replay.discounts`` when the buffer supplies it, so nothing in
the training loop changes.

The walk stops early at an episode boundary and never crosses the write head,
so a sample near either is simply shorter than ``n`` rather than mixing two
episodes together.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class NStepReplayBufferSamples(NamedTuple):
    """``ReplayBufferSamples`` plus the per-sample bootstrap discount."""

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor


class NStepReplayBuffer(ReplayBuffer):
    """Replay buffer returning ``n``-step returns and their discounts."""

    def __init__(self, *args, n_steps: int = 5, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if self.optimize_memory_usage:
            raise ValueError(
                "n-step walks read next_observations directly; "
                "optimize_memory_usage must be False")
        if not self.handle_timeout_termination:
            raise ValueError(
                "n-step returns must distinguish timeouts from falls; "
                "handle_timeout_termination must be True")
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)

    def _forward_limit(self, index: int) -> int:
        """How far a walk starting at ``index`` may run before the write head.

        ``self.pos`` is the next slot to be written, so it holds the oldest
        transition once the buffer is full and is never a valid continuation.
        """
        if not self.full:
            return self.pos - index
        return (self.pos - index) % self.buffer_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: VecNormalize | None = None) -> NStepReplayBufferSamples:
        env_indices = np.random.randint(
            0, high=self.n_envs, size=(len(batch_inds),))

        rewards = np.zeros(len(batch_inds), dtype=np.float32)
        discounts = np.zeros(len(batch_inds), dtype=np.float32)
        final_inds = np.empty(len(batch_inds), dtype=np.int64)
        dones = np.zeros(len(batch_inds), dtype=np.float32)

        for row, (start, env_index) in enumerate(zip(batch_inds, env_indices)):
            budget = min(self.n_steps, max(1, self._forward_limit(int(start))))
            index = int(start)
            discount = 1.0
            total = 0.0
            for _ in range(budget):
                reward = self._normalize_reward(
                    self.rewards[index, env_index].reshape(1, -1), env).item()
                total += discount * reward
                discount *= self.gamma
                # A timeout is not a real terminal: bootstrap through it, but
                # stop walking because the next slot begins a new episode.
                terminal = float(self.dones[index, env_index]) * (
                    1.0 - float(self.timeouts[index, env_index]))
                episode_over = bool(self.dones[index, env_index])
                final_inds[row] = index
                dones[row] = terminal
                if episode_over:
                    break
                index = (index + 1) % self.buffer_size
            rewards[row] = total
            discounts[row] = discount

        observations = self._normalize_obs(
            self.observations[batch_inds, env_indices, :], env)
        next_observations = self._normalize_obs(
            self.next_observations[final_inds, env_indices, :], env)
        return NStepReplayBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self.to_torch(next_observations),
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(rewards.reshape(-1, 1)),
            discounts=self.to_torch(discounts.reshape(-1, 1)),
        )
