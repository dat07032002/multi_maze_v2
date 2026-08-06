"""Actuator contract for the FEETECH STS3215 tilt servos.

Three layers, so hardware units never leak into the policy:

    policy action [-1, 1]  ->  desired board angle (rad)  ->  servo counts
            contract                    contract                 driver

The middle representation is physical, which is the point. The previous contract
mapped policy output straight onto a servo's native units (``servo_home=500``,
``servo_limits=(100, 900)``), so it silently described a 0-1000 device and became
wrong the moment the motor changed. Board angle is a property of the machine, not
of whatever servo happens to be bolted to it.

The angle -> counts mapping is **measured**, never assumed. A nominal STS3215 has
4096 counts over 2*pi rad (about 651.9 counts/rad), but that only equals the board
mapping if the servo drives the plate 1:1, which no real linkage does. Gain,
offset, and sign all come from the static sweep in ``tools/sysid_actuator.py``.

Until those measurements exist, ``AxisCalibration.measured`` is False and any
attempt to convert an angle to counts raises. Bring-up deliberately bypasses this
by commanding raw counts through the driver: you cannot calibrate the mapping
using the mapping.

These numbers describe **position mode** (``Register.MODE == 0``) with the servo's
own PID as recorded at measurement time. Switching to PWM mode, or retuning the
internal gains, invalidates them -- hence ``servo_config`` on the contract.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONTRACT_VERSION = "tag_sts3215_servo_v1"

# Nominal STS3215 scaling, for sanity-checking a measured fit -- not for use.
NOMINAL_COUNTS_PER_REV = 4096
NOMINAL_COUNTS_PER_RAD = NOMINAL_COUNTS_PER_REV / (2.0 * math.pi)


class CalibrationError(RuntimeError):
    """Raised when an unmeasured or out-of-range calibration is used."""


@dataclass(frozen=True)
class AxisCalibration:
    """Measured mapping from board angle to servo counts for one axis.

    counts = center_counts + sign * counts_per_rad * angle_rad
    """

    servo_id: int
    counts_per_rad: float = NOMINAL_COUNTS_PER_RAD
    center_counts: int = 2048
    sign: int = 1

    # Hard travel limits in counts. Commands are clamped to these regardless of
    # what the angle limit permits, so a bad gain cannot drive into a hard stop.
    min_counts: int = 0
    max_counts: int = 4095

    # Measured, informational: consumed by the simulator, not by the mapping.
    deadband_counts: float | None = None
    backlash_rad: float | None = None
    step_latency_s: float | None = None
    rise_time_s: float | None = None
    max_rate_rad_s: float | None = None

    measured: bool = False

    def __post_init__(self) -> None:
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {self.sign}")
        if self.counts_per_rad <= 0:
            raise ValueError("counts_per_rad must be positive; encode direction in sign")
        if not self.min_counts <= self.center_counts <= self.max_counts:
            raise ValueError(
                f"center_counts {self.center_counts} outside travel limits "
                f"{self.min_counts}..{self.max_counts}"
            )

    def angle_to_counts(self, angle_rad: float) -> int:
        """Convert a board angle to a servo count, clamped to travel limits."""
        if not self.measured:
            raise CalibrationError(
                f"axis on servo {self.servo_id} is not calibrated. Run "
                "tools/sysid_actuator.py to measure gain, offset, and sign. "
                "Bring-up should command raw counts through the driver instead."
            )
        raw = self.center_counts + self.sign * self.counts_per_rad * float(angle_rad)
        return int(round(min(max(raw, self.min_counts), self.max_counts)))

    def counts_to_angle(self, counts: float) -> float:
        """Inverse mapping, for interpreting encoder feedback as board angle."""
        if not self.measured:
            raise CalibrationError(
                f"axis on servo {self.servo_id} is not calibrated."
            )
        return (float(counts) - self.center_counts) / (self.sign * self.counts_per_rad)


@dataclass(frozen=True)
class ServoContract:
    """Policy action <-> board angle <-> servo counts, for both tilt axes."""

    roll: AxisCalibration = field(
        default_factory=lambda: AxisCalibration(servo_id=1)
    )
    pitch: AxisCalibration = field(
        default_factory=lambda: AxisCalibration(servo_id=2)
    )

    # Action [-1, 1] maps linearly onto +-max_tilt_rad. Start conservative and
    # widen only once the measured mapping and the travel limits agree.
    max_tilt_rad: float = math.radians(6.0)

    # A pose older than this is not evidence of where the board is.
    max_pose_age_s: float = 0.05

    # Servo settings in force when the calibration was measured. Recorded so a
    # later session can tell whether the numbers still apply.
    servo_config: dict = field(default_factory=dict)

    version: str = CONTRACT_VERSION

    @property
    def axes(self) -> tuple[AxisCalibration, AxisCalibration]:
        return (self.roll, self.pitch)

    @property
    def measured(self) -> bool:
        return all(axis.measured for axis in self.axes)

    # ---- action -> angle ---------------------------------------------------
    def action_to_angle(self, action) -> tuple[float, float]:
        """Map a policy action in [-1, 1]^2 to (roll, pitch) board angles.

        Non-finite input is rejected rather than clipped: NaN reaching the servos
        is a bug worth surfacing, not smoothing over.
        """
        values = tuple(float(v) for v in action)
        if len(values) != 2:
            raise ValueError(f"action must be 2 values, got {len(values)}")
        for value in values:
            if not math.isfinite(value):
                raise ValueError(f"action contains a non-finite value: {values}")
        clipped = tuple(min(max(v, -1.0), 1.0) for v in values)
        return (clipped[0] * self.max_tilt_rad, clipped[1] * self.max_tilt_rad)

    def angle_to_action(self, roll_rad: float, pitch_rad: float) -> tuple[float, float]:
        """Inverse of action_to_angle, for replaying recorded angles as actions."""
        return (
            min(max(roll_rad / self.max_tilt_rad, -1.0), 1.0),
            min(max(pitch_rad / self.max_tilt_rad, -1.0), 1.0),
        )

    # ---- angle -> counts ---------------------------------------------------
    def angles_to_counts(self, roll_rad: float, pitch_rad: float) -> tuple[int, int]:
        return (
            self.roll.angle_to_counts(roll_rad),
            self.pitch.angle_to_counts(pitch_rad),
        )

    def action_to_counts(self, action) -> tuple[int, int]:
        """Full chain, the only call the control loop needs."""
        roll_rad, pitch_rad = self.action_to_angle(action)
        return self.angles_to_counts(roll_rad, pitch_rad)

    def counts_to_angles(self, roll_counts: float, pitch_counts: float):
        return (
            self.roll.counts_to_angle(roll_counts),
            self.pitch.counts_to_angle(pitch_counts),
        )

    # ---- persistence -------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n",
                              encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ServoContract":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("version")
        if version != CONTRACT_VERSION:
            raise CalibrationError(
                f"calibration file is {version!r}, this code expects "
                f"{CONTRACT_VERSION!r}. Re-measure rather than reinterpreting "
                "old numbers under new semantics."
            )
        return cls(
            roll=AxisCalibration(**data["roll"]),
            pitch=AxisCalibration(**data["pitch"]),
            max_tilt_rad=data["max_tilt_rad"],
            max_pose_age_s=data.get("max_pose_age_s", 0.05),
            servo_config=data.get("servo_config", {}),
            version=version,
        )
