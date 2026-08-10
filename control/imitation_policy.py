"""Route-conditioned policy used for behavior cloning and deployment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from contract import policy_contract as pc


class RouteConditionedPolicy(nn.Module):
    """A compact MLP with its observation normalization embedded."""

    def __init__(self, observation_mean=None, observation_std=None,
                 hidden_sizes=(256, 256, 128)):
        super().__init__()
        mean = np.zeros(pc.OBSERVATION_SIZE, dtype=np.float32) \
            if observation_mean is None else np.asarray(
                observation_mean, dtype=np.float32)
        std = np.ones(pc.OBSERVATION_SIZE, dtype=np.float32) \
            if observation_std is None else np.asarray(
                observation_std, dtype=np.float32)
        if mean.ndim != 1 or mean.shape != std.shape:
            raise ValueError("normalization arrays must be matching vectors")
        self.observation_size = int(mean.size)
        self.register_buffer("observation_mean", torch.from_numpy(mean))
        self.register_buffer("observation_std", torch.from_numpy(
            np.maximum(std, 1e-4)))

        layers = []
        previous = self.observation_size
        for size in hidden_sizes:
            linear = nn.Linear(previous, size)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.SiLU(), nn.LayerNorm(size)])
            previous = size
        output = nn.Linear(previous, 2)
        nn.init.orthogonal_(output.weight, gain=0.01)
        nn.init.zeros_(output.bias)
        layers.extend([output, nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.observation_mean) \
            / self.observation_std
        return self.network(torch.clamp(normalized, -10.0, 10.0))

    @torch.no_grad()
    def predict(self, observation) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape[-1] != self.observation_size:
            legacy_size = pc.OBSERVATION_SIZE - pc.LOOKAHEAD_COUNT
            if (self.observation_size == legacy_size
                    and array.shape[-1] == pc.OBSERVATION_SIZE):
                # Clearance was appended to policy contract v2, so legacy
                # checkpoints consume the unchanged prefix for comparisons.
                array = array[..., :legacy_size]
            else:
                raise ValueError(
                    f"observation has {array.shape[-1]} features, policy "
                    f"expects {self.observation_size}")
        values = torch.as_tensor(
            array,
            device=self.observation_mean.device)
        single = values.ndim == 1
        if single:
            values = values.unsqueeze(0)
        action = self(values).cpu().numpy()
        return action[0] if single else action


def load_policy(path: str | Path, device="cpu") -> RouteConditionedPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = RouteConditionedPolicy(
        checkpoint["observation_mean"], checkpoint["observation_std"],
        hidden_sizes=tuple(config["hidden_sizes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model
