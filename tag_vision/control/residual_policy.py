"""Safety boundary between a model controller and a learned residual."""
from __future__ import annotations

import numpy as np


class ResidualActionMixer:
    """Bound SAC to a small correction; authority can be ramped after evidence."""

    def __init__(self, *, max_total_tilt_deg: float = 4.0,
                 max_residual_tilt_deg: float = 0.35,
                 initial_authority: float = 0.0) -> None:
        if max_total_tilt_deg <= 0 or max_residual_tilt_deg < 0:
            raise ValueError("invalid action limits")
        self.max_total_tilt_deg = float(max_total_tilt_deg)
        self.max_residual_tilt_deg = float(max_residual_tilt_deg)
        self.set_authority(initial_authority)

    def set_authority(self, authority: float) -> None:
        if not 0 <= authority <= 1:
            raise ValueError("authority must be in [0, 1]")
        self.authority = float(authority)

    def mix(self, model_tilt_deg, normalized_residual) -> np.ndarray:
        base = np.asarray(model_tilt_deg, dtype=np.float64)
        residual = np.asarray(normalized_residual, dtype=np.float64)
        if base.shape != (2,) or residual.shape != (2,):
            raise ValueError("actions must be two-vectors")
        residual = np.clip(residual, -1.0, 1.0)
        command = (base + self.authority * self.max_residual_tilt_deg
                   * residual)
        return np.clip(command, -self.max_total_tilt_deg,
                       self.max_total_tilt_deg)
