"""Direction-dependent inverse maps for a linkage with measurable backlash."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _interp_extrapolate(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    if x <= xs[0]:
        lo, hi = 0, 1
    elif x >= xs[-1]:
        lo, hi = len(xs) - 2, len(xs) - 1
    else:
        hi = next(i for i, value in enumerate(xs) if value >= x)
        lo = hi - 1
    fraction = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + fraction * (ys[hi] - ys[lo])


@dataclass(frozen=True)
class DirectionalBranch:
    """Monotonic measured physical angles and the counts that produced them."""

    angles_deg: tuple[float, ...]
    counts: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.angles_deg) != len(self.counts) or len(self.counts) < 2:
            raise ValueError("a branch needs at least two angle/count pairs")
        if any(b <= a for a, b in zip(self.angles_deg, self.angles_deg[1:])):
            raise ValueError("branch angles must be strictly increasing")

    def counts_for(self, angle_deg: float) -> float:
        return _interp_extrapolate(float(angle_deg), self.angles_deg, self.counts)


@dataclass(frozen=True)
class DirectionalAxis:
    servo_id: int
    level_counts: int
    min_counts: int
    max_counts: int
    up: DirectionalBranch
    down: DirectionalBranch

    def counts_for(self, angle_deg: float, direction: int) -> int:
        if direction == 0:
            raw = self.level_counts
        else:
            raw = (self.up if direction > 0 else self.down).counts_for(angle_deg)
        return int(round(min(max(raw, self.min_counts), self.max_counts)))


@dataclass(frozen=True)
class DirectionalMotorCalibration:
    alpha: DirectionalAxis
    beta: DirectionalAxis
    source_run: str
    # Rows are board (alpha, beta); columns are counts on the logical alpha
    # and beta servos respectively.
    jacobian_deg_per_count: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.0), (0.0, 1.0))
    version: str = "tag_directional_motor_v1"

    @classmethod
    def from_json(cls, path: str | Path) -> "DirectionalMotorCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != "tag_directional_motor_v1":
            raise ValueError("unsupported directional motor calibration")

        def axis(name: str) -> DirectionalAxis:
            item = data[name]
            branch = lambda key: DirectionalBranch(  # noqa: E731
                tuple(item[key]["angles_deg"]), tuple(item[key]["counts"]))
            return DirectionalAxis(
                servo_id=int(item["servo_id"]),
                level_counts=int(item["level_counts"]),
                min_counts=int(item["min_counts"]),
                max_counts=int(item["max_counts"]),
                up=branch("up"), down=branch("down"),
            )

        jacobian = data.get("jacobian_deg_per_count", ((1.0, 0.0),
                                                        (0.0, 1.0)))
        return cls(
            alpha=axis("alpha"), beta=axis("beta"),
            source_run=data["source_run"],
            jacobian_deg_per_count=tuple(
                tuple(float(value) for value in row) for row in jacobian),
            version=data["version"],
        )


class DirectionalMotorOrigin:
    """Stateful target mapper selecting a branch from logical travel direction."""

    def __init__(self, calibration: DirectionalMotorCalibration) -> None:
        self.calibration = calibration
        self.last_targets = [0.0, 0.0]
        self.directions = [0, 0]

    def targets(self, alpha_deg: float, beta_deg: float) -> dict[int, int]:
        requested = [float(alpha_deg), float(beta_deg)]
        result = {}
        for index, (axis, target) in enumerate(zip(
                (self.calibration.alpha, self.calibration.beta), requested)):
            previous = self.last_targets[index]
            if target > previous:
                self.directions[index] = 1
            elif target < previous:
                self.directions[index] = -1
            result[axis.servo_id] = axis.counts_for(
                target, self.directions[index])
            self.last_targets[index] = target
        return result
