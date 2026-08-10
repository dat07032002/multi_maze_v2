"""Small behaviour-cloning helper shared by M3 and M4."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


def pretrain_actor(model: SAC, observations: np.ndarray,
                   actions: np.ndarray, epochs: int,
                   batch_size: int = 1024) -> None:
    """Fit SAC's mean action and initialise exploration near one quantum."""
    dataset = TensorDataset(torch.from_numpy(observations),
                            torch.from_numpy(actions))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    actor = model.policy.actor
    actor.train()
    for epoch in range(epochs):
        losses = []
        for obs, target in loader:
            obs = obs.to(model.device)
            target = target.to(model.device)
            mean, log_std, _ = actor.get_action_dist_params(obs)
            predicted = torch.tanh(mean)
            mean_loss = F.mse_loss(predicted, target)
            std_loss = F.mse_loss(log_std, torch.full_like(log_std, -3.0))
            loss = mean_loss + 0.05 * std_loss
            actor.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor.optimizer.step()
            losses.append(float(mean_loss.detach().cpu()))
        print(f"BC epoch {epoch + 1:02d}/{epochs}: "
              f"action_mse={np.mean(losses):.6f}")


def seed_replay_buffer(model: SAC, transitions: dict[str, np.ndarray]) -> int:
    """Insert valid same-task demonstrations into an n-env replay buffer."""
    n_envs = model.n_envs
    count = len(transitions["observations"])
    usable = count - count % n_envs
    for start in range(0, usable, n_envs):
        stop = start + n_envs
        truncated = transitions["truncated"][start:stop]
        infos = [{"TimeLimit.truncated": bool(value)}
                 for value in truncated]
        model.replay_buffer.add(
            transitions["observations"][start:stop],
            transitions["next_observations"][start:stop],
            transitions["actions"][start:stop],
            transitions["rewards"][start:stop],
            transitions["dones"][start:stop], infos)
    print(f"seeded replay buffer with {usable:,} same-task transitions")
    return usable


def warmup_critics(model: SAC, gradient_steps: int,
                   batch_size: int = 512) -> None:
    """Fit M4 critics to seeded demonstrations without moving the actor."""
    model.policy.set_training_mode(True)
    losses = []
    for step in range(gradient_steps):
        replay = model.replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_actions = model.actor(replay.next_observations,
                                       deterministic=True)
            next_q = torch.cat(model.critic_target(
                replay.next_observations, next_actions), dim=1)
            next_q, _ = torch.min(next_q, dim=1, keepdim=True)
            target = (replay.rewards
                      + (1.0 - replay.dones) * model.gamma * next_q)
        current = model.critic(replay.observations, replay.actions)
        loss = 0.5 * sum(F.mse_loss(value, target) for value in current)
        model.critic.optimizer.zero_grad()
        loss.backward()
        model.critic.optimizer.step()
        if step % model.target_update_interval == 0:
            polyak_update(model.critic.parameters(),
                          model.critic_target.parameters(), model.tau)
        losses.append(float(loss.detach().cpu()))
    print(f"critic warmup {gradient_steps:,} steps: "
          f"loss {np.mean(losses[-100:]):.6f}")
