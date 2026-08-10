"""Temporally coherent stochastic actions for real-system exploration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExplorationAction:
    target_deg: np.ndarray
    segment: int


class StuckRecovery:
    """Detect sustained lack of motion before falling back to exploration."""

    def __init__(self, *, speed_threshold_mm_s: float = 3.0,
                 duration_s: float = 3.0) -> None:
        if speed_threshold_mm_s < 0 or duration_s <= 0:
            raise ValueError("invalid stuck recovery thresholds")
        self.speed_threshold_mm_s = float(speed_threshold_mm_s)
        self.duration_s = float(duration_s)
        self._since: float | None = None

    def reset(self) -> None:
        self._since = None

    def update(self, speed_mm_s: float, now_s: float) -> bool:
        if float(speed_mm_s) >= self.speed_threshold_mm_s:
            self._since = None
            return False
        now = float(now_s)
        if self._since is None:
            self._since = now
        return now - self._since >= self.duration_s


def episode_policy(*, episode: int, learned_model_ready: bool,
                   explore_every: int = 5) -> str:
    """Choose random data collection or learned-model control for an episode.

    Fresh runs must explore until their first model exists.  Afterwards a
    deterministic cadence makes the data mix auditable and guarantees that
    online training continues to see states outside the current planner's
    preferred trajectories.  ``explore_every=0`` disables periodic episodes.
    """
    if episode < 1 or explore_every < 0:
        raise ValueError("invalid episode exploration schedule")
    if not learned_model_ready:
        return "explore"
    if explore_every and episode % explore_every == 0:
        return "explore"
    return "cem"


class SmoothRandomExploration:
    """Sample full-range targets and hold them across several control steps.

    The independent safety shield applies the actual slew and absolute limits.
    Holding a target avoids high-frequency random chatter and lets static
    friction, motor lag, and ball acceleration become observable in replay.
    """

    def __init__(self, *, max_tilt_deg: float = 4.0,
                 hold_s: float = 0.8, seed: int = 0) -> None:
        if max_tilt_deg <= 0 or hold_s <= 0:
            raise ValueError("exploration limits must be positive")
        self.max_tilt_deg = float(max_tilt_deg)
        self.hold_s = float(hold_s)
        self.rng = np.random.default_rng(seed)
        self._target = np.zeros(2, dtype=np.float64)
        self._until: float | None = None
        self._segment = 0

    def reset(self) -> None:
        self._target[:] = 0.0
        self._until = None
        self._segment = 0

    def command(self, now_s: float) -> ExplorationAction:
        now = float(now_s)
        if self._until is None or now >= self._until:
            self._target = self.rng.uniform(
                -self.max_tilt_deg, self.max_tilt_deg, size=2)
            self._until = now + self.hold_s
            self._segment += 1
        return ExplorationAction(self._target.copy(), self._segment)
