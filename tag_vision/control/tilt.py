"""Commanded board angle -> servo counts, with the lost motion fed forward.

This is the layer between a policy and the driver. It is deliberately thin, and
deliberately *not* a PID.

Why not feedback. About 100 ms of the step latency is electrical and
irreducible, which caps a stable feedback loop somewhere near 0.2-0.3 Hz -- too
slow to help a policy that is already reacting. Meanwhile the two dominant
errors are both deterministic and both measured: the counts/angle gain, and
~1.35 deg of lost motion on reversal. Feedback is for what you cannot predict;
spending it on things already in a calibration file is the wrong tool. An
integrator would be actively harmful, winding up while the plate sits inside the
backlash band and overshooting when the slack finally takes up.

Why the backlash term is the whole point. Measured on this rig:

    IMU noise                    0.006 deg
    command resolution           0.19 deg   (40 counts)
    linear model residual        0.23 deg
    backlash, unconditioned      ~1.35 deg  <- seven times the rest combined

The plate's angle at a given count depends on which side of the slack it is
resting against, so the same command lands ~1.35 deg apart depending on the
direction of travel. Compensating for it is worth more than every other
refinement here put together.

The model is the standard backlash inverse, referenced to whichever face of the
band the calibration was fitted on. ``sysid_actuator`` conditions every point by
approaching from below, so the fit is the **up** branch::

    travelling up:    plate = g*(counts - centre)
    travelling down:  plate = g*(counts - centre) + B

To place the plate at ``theta`` you therefore aim at ``theta`` going up and
``theta - B`` going down -- asymmetric, not +-B/2. Assuming the fit was the band
*centre* and applying +-B/2 puts every command half a backlash high; measured on
this rig that was +0.85 deg on roll against a B/2 of 0.66. ``AxisBacklash.reference``
carries which convention applies.

Direction comes from the requested change, and is held across a no-op request
rather than reset, because "no change" does not move the plate off whichever
face it is resting on.

A consequence worth stating plainly: reversing direction costs a servo move of
at least ``B*g`` counts (about 277 on roll, 269 on pitch) before the plate
begins to move at all. That is not this code being clumsy, it is the mechanism.
Small dithering commands around a setpoint will therefore produce large servo
motion and little plate motion, which is a good reason for a policy to avoid
them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..hardware.sts3215 import Register, STS3215Bus


@dataclass
class AxisBacklash:
    """Lost motion for one axis, in radians of board angle.

    ``reference`` says which face of the backlash band the *calibration* sits
    on, and getting it wrong is a silent half-backlash offset.

    ``sysid_actuator`` conditions every sweep point by driving below the target
    and coming up, so the fit describes the **up** branch -- the lower edge of
    the band -- not its middle::

                        <- fitted here (conditioned, approached from below)
        up branch    ---+--------------
                        |  B
        down branch  ---+--------------

    With ``reference="up"`` the inverse is therefore asymmetric: nothing when
    travelling up, minus the full B when travelling down. Treating the fit as
    the band centre and applying +-B/2 instead put every command half a
    backlash high -- measured at +0.85 deg on roll, where B/2 is 0.66.

    ``measured`` distinguishes a real figure from the zero default. Running
    unmeasured is legitimate -- it degrades to plain feedforward -- but it
    should be a choice rather than an accident.
    """

    lost_motion_rad: float = 0.0
    measured: bool = False
    reference: str = "up"

    def __post_init__(self) -> None:
        if self.reference not in ("up", "centre", "down"):
            raise ValueError(
                f"reference must be 'up', 'centre' or 'down', got "
                f"{self.reference!r}")

    def offset_for(self, direction: int) -> float:
        """Angle to add to a request when travelling in ``direction``."""
        if direction == 0:
            return 0.0
        B = self.lost_motion_rad
        if self.reference == "centre":
            return direction * B / 2.0
        if self.reference == "up":
            # Calibration is the up branch: going up needs nothing, going down
            # must first give away the whole band.
            return 0.0 if direction > 0 else -B
        return B if direction > 0 else 0.0  # reference == "down"


class BacklashCompensator:
    """Tracks which face of the slack the axis is resting on."""

    def __init__(self, backlash: AxisBacklash) -> None:
        self.backlash = backlash
        # +1 last travelled up, -1 down, 0 unknown. Starting unknown matters:
        # until the axis has moved we cannot know which face it is against, so
        # the first command applies no correction rather than guessing and
        # being wrong half the time.
        self.direction = 0
        self.last_target_rad: float | None = None
        self.reversals = 0

    def reset(self) -> None:
        self.direction = 0
        self.last_target_rad = None

    def target_for(self, angle_rad: float) -> float:
        """Angle to hand the inverse mapping so the plate lands on ``angle_rad``."""
        previous = self.last_target_rad
        if previous is None:
            new_direction = 0
        elif angle_rad > previous:
            new_direction = 1
        elif angle_rad < previous:
            new_direction = -1
        else:
            # Unchanged request: the plate has not left the face it is on.
            new_direction = self.direction

        if new_direction != 0 and self.direction != 0 and new_direction != self.direction:
            self.reversals += 1

        self.last_target_rad = angle_rad
        self.direction = new_direction
        return angle_rad + self.backlash.offset_for(new_direction)


@dataclass
class TiltLimits:
    """Hard count bounds per servo, independent of the calibrated range.

    These come from ``calib/servo_travel_limits.json`` and are wider than the
    range the angle mapping was fitted over. They exist to stop a bad command
    driving into a stop, not to say where the mapping is trustworthy.
    """

    min_counts: dict[int, int]
    max_counts: dict[int, int]

    @classmethod
    def from_json(cls, path: str | Path) -> "TiltLimits":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lo, hi = {}, {}
        for axis in ("roll", "pitch"):
            entry = data[axis]
            lo[entry["servo_id"]] = int(entry["safe"]["min_counts"])
            hi[entry["servo_id"]] = int(entry["safe"]["max_counts"])
        return cls(min_counts=lo, max_counts=hi)

    def clamp(self, servo_id: int, counts: int) -> tuple[int, bool]:
        lo = self.min_counts.get(servo_id, 0)
        hi = self.max_counts.get(servo_id, 4095)
        clamped = int(min(max(counts, lo), hi))
        return clamped, clamped != counts


@dataclass
class TiltCommand:
    """What a single command actually did, for logging and for the caller."""

    requested_rad: tuple[float, float]
    compensated_rad: tuple[float, float]
    counts: tuple[int, int]
    clamped: bool
    reversed_axes: tuple[bool, bool]


class TiltController:
    """Feedforward tilt commands with backlash compensation and hard limits."""

    def __init__(
        self,
        bus: STS3215Bus,
        contract,
        limits: TiltLimits | None = None,
        roll_backlash: AxisBacklash | None = None,
        pitch_backlash: AxisBacklash | None = None,
    ) -> None:
        if not contract.measured:
            raise ValueError(
                "contract is not measured; angle_to_counts would raise. Run "
                "tools/sysid_actuator.py and tools/fit_sysid.py first.")
        self.bus = bus
        self.contract = contract
        self.limits = limits
        self.roll = BacklashCompensator(roll_backlash or AxisBacklash())
        self.pitch = BacklashCompensator(pitch_backlash or AxisBacklash())
        self.last: TiltCommand | None = None

    @property
    def compensating(self) -> bool:
        return self.roll.backlash.measured and self.pitch.backlash.measured

    def reset(self) -> None:
        """Forget the direction state, e.g. after the board has been moved."""
        self.roll.reset()
        self.pitch.reset()

    # ---- command paths -----------------------------------------------------
    def counts_for_angles(self, roll_rad: float, pitch_rad: float) -> TiltCommand:
        """Resolve angles to counts without touching the bus."""
        before = (self.roll.direction, self.pitch.direction)
        aimed = (self.roll.target_for(roll_rad), self.pitch.target_for(pitch_rad))

        raw = self.contract.angles_to_counts(*aimed)
        counts, clamped = [], False
        for axis, value in zip(self.contract.axes, raw):
            if self.limits is None:
                counts.append(value)
                continue
            fixed, hit = self.limits.clamp(axis.servo_id, value)
            counts.append(fixed)
            clamped = clamped or hit

        after = (self.roll.direction, self.pitch.direction)
        reversed_axes = tuple(
            b != 0 and a != 0 and b != a for a, b in zip(before, after))

        command = TiltCommand(
            requested_rad=(roll_rad, pitch_rad),
            compensated_rad=aimed,
            counts=(counts[0], counts[1]),
            clamped=clamped,
            reversed_axes=reversed_axes,  # type: ignore[arg-type]
        )
        self.last = command
        return command

    def command_angles(self, roll_rad: float, pitch_rad: float) -> TiltCommand:
        """Drive both axes to the given board angles."""
        command = self.counts_for_angles(roll_rad, pitch_rad)
        targets = {
            axis.servo_id: value
            for axis, value in zip(self.contract.axes, command.counts)
        }
        # One packet, so both axes latch together rather than a bus round trip
        # apart.
        self.bus.sync_write_positions(targets)
        return command

    def command_action(self, action) -> TiltCommand:
        """Policy action in [-1, 1]^2 -> board angles -> counts."""
        roll_rad, pitch_rad = self.contract.action_to_angle(action)
        return self.command_angles(roll_rad, pitch_rad)

    # ---- lifecycle ---------------------------------------------------------
    def enable(self) -> None:
        for axis in self.contract.axes:
            self.bus.torque_enable(axis.servo_id)

    def release(self) -> None:
        for axis in self.contract.axes:
            self.bus.torque_disable(axis.servo_id)

    def read_counts(self) -> dict[int, int]:
        return {axis.servo_id: self.bus.read_word(axis.servo_id,
                                                  Register.PRESENT_POSITION)
                for axis in self.contract.axes}


def load_backlash(path: str | Path) -> tuple[AxisBacklash, AxisBacklash]:
    """Read per-axis lost motion from a calibration file.

    Prefers an explicit ``backlash_unconditioned_rad`` if present, because that
    is the figure a reversing controller experiences; ``backlash_rad`` from a
    conditioned sweep describes calibration conditions and is roughly an order
    of magnitude smaller.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for axis in ("roll", "pitch"):
        entry = data.get(axis, {})
        value = entry.get("backlash_unconditioned_rad")
        if value is None:
            value = entry.get("backlash_rad")
        out.append(AxisBacklash(lost_motion_rad=float(value or 0.0),
                                measured=value is not None))
    return out[0], out[1]


def degrees(value_rad: float) -> float:
    return math.degrees(value_rad)
