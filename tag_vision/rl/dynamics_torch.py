"""Probabilistic neural dynamics ensemble for local-GPU model-based RL.

PyTorch is optional for the rest of the project. Importing this module remains
safe without it; constructing an ensemble gives one actionable error.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on hardware runtime setup
    torch = None
    nn = None


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install the CUDA build only after "
            "tools/rl_preflight.py reports a healthy NVIDIA driver.")
    return torch


def motion_balanced_bootstrap(replay, indices, rng,
                              *, minimum_speed_mm_s: float = 3.0):
    """Bootstrap equally from moving and stationary real transitions."""
    selected = np.asarray(indices, dtype=np.int64)
    if replay.observation_size < 4 or len(selected) < 2:
        return rng.choice(selected, size=len(selected), replace=True)
    before = np.linalg.norm(replay.observations[selected, 2:4], axis=1)
    after = np.linalg.norm(replay.next_observations[selected, 2:4], axis=1)
    moving_mask = np.maximum(before, after) >= float(minimum_speed_mm_s)
    moving = selected[moving_mask]
    stationary = selected[~moving_mask]
    if not len(moving) or not len(stationary):
        return rng.choice(selected, size=len(selected), replace=True)
    moving_size = len(selected) // 2
    bootstrapped = np.concatenate((
        rng.choice(moving, size=moving_size, replace=True),
        rng.choice(stationary, size=len(selected) - moving_size, replace=True),
    ))
    rng.shuffle(bootstrapped)
    return bootstrapped


if nn is not None:
    class ProbabilisticMLP(nn.Module):
        def __init__(self, input_size: int, output_size: int,
                     hidden_size: int = 256) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, 2 * output_size),
            )
            self.output_size = output_size

        def forward(self, values):
            mean, raw_log_variance = self.network(values).split(
                self.output_size, dim=-1)
            # Smoothly bounded variance prevents numerical collapse and giant
            # uncertainty from poisoning CEM before much real data exists.
            log_variance = -10.0 + 11.0 * torch.sigmoid(raw_log_variance)
            return mean, log_variance


    class EnsembleDynamics(nn.Module):
        def __init__(self, observation_size: int, action_size: int = 2, *,
                     members: int = 5, hidden_size: int = 256,
                     device: str = "cuda") -> None:
            super().__init__()
            if members < 2:
                raise ValueError("dynamics uncertainty needs at least two members")
            self.observation_size = int(observation_size)
            self.action_size = int(action_size)
            self.member_count = int(members)
            self.models = nn.ModuleList([
                ProbabilisticMLP(observation_size + action_size,
                                 observation_size, hidden_size)
                for _ in range(members)
            ])
            self.register_buffer("input_mean", torch.zeros(
                observation_size + action_size))
            self.register_buffer("input_std", torch.ones(
                observation_size + action_size))
            self.register_buffer("delta_mean", torch.zeros(observation_size))
            self.register_buffer("delta_std", torch.ones(observation_size))
            self.device_name = str(device)
            self.to(torch.device(device))

        @property
        def device(self):
            return self.input_mean.device

        def set_normalization(self, observations, actions, next_observations) -> None:
            joined = torch.cat((observations, actions), dim=-1)
            delta = next_observations - observations
            self.input_mean.copy_(joined.mean(dim=0))
            self.input_std.copy_(joined.std(dim=0).clamp_min(1e-4))
            self.delta_mean.copy_(delta.mean(dim=0))
            self.delta_std.copy_(delta.std(dim=0).clamp_min(1e-4))

        def _model_delta(self, model, observations, actions):
            joined = torch.cat((observations, actions), dim=-1)
            normalized = (joined - self.input_mean) / self.input_std
            mean, log_variance = model(normalized)
            physical_mean = mean * self.delta_std + self.delta_mean
            physical_variance = log_variance.exp() * self.delta_std.square()
            return physical_mean, physical_variance

        @torch.no_grad()
        def predict_members(self, observations, actions):
            """One shared candidate batch -> [ensemble, candidates, state]."""
            means, variances = [], []
            for model in self.models:
                delta, variance = self._model_delta(model, observations, actions)
                means.append(observations + delta)
                variances.append(variance)
            return torch.stack(means), torch.stack(variances)

        @torch.no_grad()
        def predict_member_states(self, states, actions):
            """Each ensemble member advances its own candidate states."""
            next_states, variances = [], []
            for index, model in enumerate(self.models):
                delta, variance = self._model_delta(
                    model, states[index], actions)
                next_states.append(states[index] + delta)
                variances.append(variance)
            return torch.stack(next_states), torch.stack(variances)

        def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            torch.save({
                "version": "tag_ensemble_dynamics_v1",
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "member_count": self.member_count,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            }, temporary)
            temporary.replace(target)

        @classmethod
        def load_checkpoint(cls, path: str | Path, *, device: str = "cuda"):
            """Restore an ensemble and return it together with saved metadata."""
            checkpoint = torch.load(
                Path(path), map_location=torch.device(device), weights_only=True)
            if checkpoint.get("version") != "tag_ensemble_dynamics_v1":
                raise ValueError("unsupported dynamics checkpoint")
            state = checkpoint["state_dict"]
            first_weight = state.get("models.0.network.0.weight")
            if first_weight is None or first_weight.ndim != 2:
                raise ValueError("checkpoint has no valid first ensemble layer")
            ensemble = cls(
                int(checkpoint["observation_size"]),
                int(checkpoint["action_size"]),
                members=int(checkpoint["member_count"]),
                hidden_size=int(first_weight.shape[0]),
                device=device,
            )
            ensemble.load_state_dict(state, strict=True)
            ensemble.eval()
            return ensemble, dict(checkpoint.get("metadata", {}))

else:
    class EnsembleDynamics:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


class EnsembleTrainer:
    def __init__(self, ensemble: "EnsembleDynamics", *,
                 learning_rate: float = 3e-4, weight_decay: float = 1e-5) -> None:
        require_torch()
        self.ensemble = ensemble
        self.optimizers = [torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for model in ensemble.models]

    def fit(self, replay, *, epochs: int = 20, batch_size: int = 256,
            validation_fraction: float = 0.15, seed: int = 0) -> dict:
        if replay.size < max(batch_size, 100):
            raise ValueError("not enough real transitions to train dynamics")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(replay.size)
        validation_size = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices = indices[:validation_size]
        training_indices = indices[validation_size:]
        device = self.ensemble.device

        def tensor(array, selected):
            return torch.as_tensor(array[selected], device=device,
                                   dtype=torch.float32)

        all_observations = tensor(replay.observations, indices)
        all_actions = tensor(replay.actions, indices)
        all_next = tensor(replay.next_observations, indices)
        self.ensemble.set_normalization(all_observations, all_actions, all_next)
        losses = []
        self.ensemble.train()
        for _ in range(int(epochs)):
            epoch_losses = []
            for model, optimizer in zip(self.ensemble.models, self.optimizers):
                bootstrapped = motion_balanced_bootstrap(
                    replay, training_indices, rng)
                for start in range(0, len(bootstrapped), batch_size):
                    selected = bootstrapped[start:start + batch_size]
                    observation = tensor(replay.observations, selected)
                    action = tensor(replay.actions, selected)
                    target = tensor(replay.next_observations, selected) - observation
                    normalized_target = (
                        target - self.ensemble.delta_mean) / self.ensemble.delta_std
                    joined = torch.cat((observation, action), dim=-1)
                    normalized_input = (
                        joined - self.ensemble.input_mean) / self.ensemble.input_std
                    mean, log_variance = model(normalized_input)
                    loss = 0.5 * torch.mean(
                        log_variance + (normalized_target - mean).square()
                        * torch.exp(-log_variance))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                    epoch_losses.append(float(loss.detach()))
            losses.append(float(np.mean(epoch_losses)))

        self.ensemble.eval()
        with torch.no_grad():
            observation = tensor(replay.observations, validation_indices)
            action = tensor(replay.actions, validation_indices)
            target = tensor(replay.next_observations, validation_indices)
            predicted, _ = self.ensemble.predict_members(observation, action)
            error = predicted - target.unsqueeze(0)
            validation_rmse = float(torch.sqrt(torch.mean(error.square())))
            disagreement = float(torch.mean(torch.std(predicted, dim=0)))
        return {
            "training_nll": losses[-1],
            "validation_rmse": validation_rmse,
            "validation_disagreement": disagreement,
            "epochs": int(epochs),
            "training_samples": len(training_indices),
            "training_moving_samples": int(np.count_nonzero(np.maximum(
                np.linalg.norm(replay.observations[training_indices, 2:4], axis=1),
                np.linalg.norm(
                    replay.next_observations[training_indices, 2:4], axis=1))
                >= 3.0)),
            "validation_samples": len(validation_indices),
        }
