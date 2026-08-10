"""CUDA CEM planner over the learned dynamics ensemble."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics_torch import require_torch


@dataclass(frozen=True)
class CEMAction:
    action_deg: np.ndarray
    cost: float
    uncertainty: float


class TorchCEMPlanner:
    def __init__(self, ensemble, observation_spec, *, horizon: int = 10,
                 candidates: int = 512, iterations: int = 5,
                 elite_fraction: float = 0.10, max_tilt_deg: float = 4.0,
                 max_delta_deg: float = 0.5,
                 uncertainty_weight: float = 1.0,
                 action_weight: float = 0.0005) -> None:
        torch = require_torch()
        if horizon < 2 or candidates < 16 or iterations < 1:
            raise ValueError("invalid CEM dimensions")
        self.torch = torch
        self.ensemble = ensemble
        self.spec = observation_spec
        self.horizon = int(horizon)
        self.candidates = int(candidates)
        self.iterations = int(iterations)
        self.elites = max(2, int(round(candidates * elite_fraction)))
        self.max_tilt_deg = float(max_tilt_deg)
        self.max_delta_deg = float(max_delta_deg)
        self.uncertainty_weight = float(uncertainty_weight)
        self.action_weight = float(action_weight)
        self._mean = torch.zeros((horizon, 2), device=ensemble.device)

    def reset(self) -> None:
        self._mean.zero_()

    def _bounded_actions(self, raw, previous):
        actions = []
        current = previous.expand(raw.shape[0], -1)
        for step in range(self.horizon):
            desired = raw[:, step].clamp(-self.max_tilt_deg,
                                         self.max_tilt_deg)
            current = current + (desired - current).clamp(
                -self.max_delta_deg, self.max_delta_deg)
            actions.append(current)
        return self.torch.stack(actions, dim=1)

    def _rollout_cost(self, initial, actions):
        torch = self.torch
        states = initial.expand(self.ensemble.member_count,
                                self.candidates, -1).clone()
        total = torch.zeros((self.ensemble.member_count, self.candidates),
                            device=self.ensemble.device)
        uncertainty_total = torch.zeros(self.candidates,
                                        device=self.ensemble.device)
        progress_i = self.spec.index("progress")
        cross_i = self.spec.index("cross_track")
        clearance_i = self.spec.index("clearance")
        vx_i, vy_i = self.spec.index("vx"), self.spec.index("vy")
        previous_progress = states[:, :, progress_i]
        for step in range(self.horizon):
            states, aleatoric = self.ensemble.predict_member_states(
                states, actions[:, step])
            progress = states[:, :, progress_i]
            cross = states[:, :, cross_i].abs()
            clearance = states[:, :, clearance_i]
            speed2 = states[:, :, vx_i].square() + states[:, :, vy_i].square()
            total -= (progress - previous_progress) / 5.0
            total += 0.015 * cross
            total += 0.002 * torch.relu(12.0 - clearance).square()
            total += 0.00015 * speed2 * (clearance < 12.0)
            total += self.action_weight * actions[:, step].square().sum(dim=-1)
            epistemic = torch.var(states, dim=0).mean(dim=-1)
            uncertainty_total += epistemic
            total += self.uncertainty_weight * epistemic.unsqueeze(0)
            total += 0.05 * aleatoric.mean(dim=-1)
            previous_progress = progress
        return total.mean(dim=0), uncertainty_total / self.horizon

    def command(self, observation, previous_action_deg=(0.0, 0.0)) -> CEMAction:
        torch = self.torch
        initial = torch.as_tensor(observation, device=self.ensemble.device,
                                  dtype=torch.float32).reshape(1, -1)
        previous = torch.as_tensor(previous_action_deg,
                                   device=self.ensemble.device,
                                   dtype=torch.float32).reshape(1, 2)
        mean = torch.cat((self._mean[1:], self._mean[-1:]), dim=0)
        std = torch.full_like(mean, 1.5)
        best_action = torch.zeros(2, device=self.ensemble.device)
        best_cost = float("inf")
        best_uncertainty = float("inf")
        with torch.no_grad():
            for _ in range(self.iterations):
                raw = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                    (self.candidates, self.horizon, 2),
                    device=self.ensemble.device)
                actions = self._bounded_actions(raw, previous)
                costs, uncertainty = self._rollout_cost(initial, actions)
                elite_indices = torch.topk(costs, self.elites,
                                           largest=False).indices
                elites = actions[elite_indices]
                mean = elites.mean(dim=0)
                std = elites.std(dim=0).clamp_min(0.08)
                winner = int(torch.argmin(costs))
                if float(costs[winner]) < best_cost:
                    best_cost = float(costs[winner])
                    best_action = actions[winner, 0].clone()
                    best_uncertainty = float(uncertainty[winner])
        self._mean.copy_(mean)
        return CEMAction(best_action.cpu().numpy(), best_cost,
                         best_uncertainty)
