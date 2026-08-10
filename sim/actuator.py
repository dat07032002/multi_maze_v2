"""Commanded board angle -> realised board angle, on the measured chain.

The whole point of this module is that a policy trained against it meets no
surprises on the bench. It reproduces, in physical order:

    request -> tilt.py compensator -> counts -> [ dead time -> servo lag and
    rate limit -> linkage backlash ] -> plate

The bracketed part is the plant. Everything before it is the control path the
hardware will also run, imported from ``tag_vision.control.tilt`` rather than
reimplemented, so the two cannot drift.

Three things here are not what the milestone plan first assumed, each for a
reason worth keeping.

**Backlash goes last, after the lag, not first.** The slack is in the linkage
between servo horn and plate, so the shaft moves smoothly and the plate picks up
only once the slack is taken up. Putting backlash ahead of the lag would make a
reversal cost nothing extra, when in fact it costs a full ``B * counts_per_rad``
of servo travel -- about 277 counts on roll -- before the plate moves at all.
That delay-on-reversal is the single most important thing this model has to get
right, because it is what the policy must learn not to provoke.

**The recorded ``step_latency_s`` is not the dead time.** ``sysid_actuator``
defines latency as the **5 %** crossing minus half the link round trip, so a
first-order response spends ``tau * ln(1/0.95)`` of it simply climbing to 5 %.
Feeding 0.18494 s in as pure dead time would put the modelled 5 % crossing at
0.1909 s -- 3.2 % late, inside the acceptance tolerance and wrong for a knowable
reason. The pure dead times are 0.17896 s roll and 0.14495 s pitch, and the
model reproduces the recorded numbers exactly.

**The slew ceiling is a guess wearing a measurement's clothes.** ``max_rate`` is
``max()`` of the sample-to-sample rate during the step runs, so it says the plate
*was seen to reach* 6.9 deg/s -- a floor on capability, not a ceiling -- and
max() of a differentiated 200 Hz signal is biased high besides. sysid only ever
stepped 0.195-0.584 deg, where a rate limit is invisible; a policy commands up to
8 deg swings, where it is worth 275 ms on a 4 deg step. ``slew_limit_scale``
exists so this is randomised rather than asserted, and it is the one parameter in
this module that a single 10-minute rig measurement would settle.

**Command quantisation is one count, plus a deadband -- not a 40-count grid.**
The servo accepts any integer count; 40 counts is the smallest *change* that
reliably keeps its sign, which is a property of increments and stiction, not of
the position grid. Snapping absolute positions to a 40-count lattice would
invent a coarseness the hardware does not have. What is real is the measured
per-axis deadband (6.4 counts roll, 27.9 pitch): a change smaller than that does
not move the shaft. The 40-count figure keeps its job elsewhere, as the scale of
the action-rate penalty in the reward.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from contract.servo_contract import ServoContract
from tag_vision.control.tilt import AxisBacklash, BacklashCompensator

from .mjcf_builder import REPO, load_parameters

DEFAULT_CALIBRATION = REPO / "calib" / "servo_calibration.json"

# t90 - t10 for a first order lag is tau*ln(9); the 5% crossing is tau*ln(1/0.95).
_RISE_TO_TAU = math.log(9.0)
_FIVE_PERCENT = math.log(1.0 / 0.95)


@dataclass(frozen=True)
class AxisDynamics:
    """One axis of the plant, in board-angle terms."""

    dead_time_s: float
    tau_s: float
    max_rate_rad_s: float
    backlash_rad: float
    deadband_counts: float
    centre_bias_rad: float = 0.0

    @classmethod
    def from_measurements(cls, step_latency_s: float, rise_time_s: float,
                          max_rate_rad_s: float, backlash_rad: float,
                          deadband_counts: float,
                          centre_bias_rad: float = 0.0) -> "AxisDynamics":
        """Recover the model constants from what sysid actually reports."""
        tau = rise_time_s / _RISE_TO_TAU
        return cls(
            dead_time_s=max(0.0, step_latency_s - tau * _FIVE_PERCENT),
            tau_s=tau,
            max_rate_rad_s=max_rate_rad_s,
            backlash_rad=backlash_rad,
            deadband_counts=deadband_counts,
            centre_bias_rad=centre_bias_rad,
        )


class AxisPlant:
    """Dead time, first-order lag with a rate limit, then linkage backlash."""

    def __init__(self, dynamics: AxisDynamics, axis, timestep: float):
        self.dyn = dynamics
        self.axis = axis          # contract AxisCalibration, for counts <-> angle
        self.dt = timestep
        self._delay_steps = int(round(dynamics.dead_time_s / timestep))
        self.reset()

    def reset(self, angle_rad: float = 0.0) -> None:
        counts = self.axis.angle_to_counts(angle_rad)
        self.commanded_counts = counts
        self.queue: deque[int] = deque([counts] * self._delay_steps)
        self.shaft_rad = self._shaft_target(counts)
        self.plate_rad = self.shaft_rad

    def _shaft_target(self, counts: int) -> float:
        """Counts to shaft angle, including the constant centre offset.

        The bias belongs here rather than on the output: it has to be part of
        the stored state, or ``plate_rad`` and whatever ``step`` returns drift
        apart and every reading of the model depends on which one you happened
        to use.
        """
        return self.axis.counts_to_angle(counts) + self.dyn.centre_bias_rad

    def step(self, counts: int) -> float:
        # Deadband: a change too small to overcome stiction leaves the shaft
        # where it is. Measured per axis; pitch's is four times roll's.
        if abs(counts - self.commanded_counts) >= self.dyn.deadband_counts:
            self.commanded_counts = int(counts)

        self.queue.append(self.commanded_counts)
        delayed = self.queue.popleft() if self._delay_steps else self.commanded_counts

        target = self._shaft_target(delayed)
        blend = 1.0 - math.exp(-self.dt / self.dyn.tau_s)
        move = (target - self.shaft_rad) * blend
        limit = self.dyn.max_rate_rad_s * self.dt
        self.shaft_rad += max(-limit, min(limit, move))

        # Backlash referenced to the up branch, matching the calibration:
        # travelling up the plate follows the shaft, travelling down it holds
        # until the shaft has given away the whole band.
        band = self.dyn.backlash_rad
        self.plate_rad = min(max(self.plate_rad, self.shaft_rad),
                             self.shaft_rad + band)
        return self.plate_rad


class ActuatorModel:
    """Both axes, with the hardware's own backlash compensator on top."""

    def __init__(self, timestep: float, params: dict | None = None,
                 calibration: str | Path = DEFAULT_CALIBRATION,
                 compensate: bool = True):
        params = params if params is not None else load_parameters()
        self.contract = ServoContract.from_json(calibration)
        self.dt = timestep
        self.compensate = compensate

        backlash = params["actuator.backlash_operational"]
        bias = params["actuator.centre_bias"]
        self.dynamics = {
            name: AxisDynamics.from_measurements(
                params[f"actuator.{name}.step_latency"],
                params[f"actuator.{name}.rise_time"],
                params[f"actuator.{name}.max_rate"]
                * params["actuator.slew_limit_scale"],
                backlash,
                getattr(self.contract, name).deadband_counts or 0.0,
                bias[index])
            for index, name in enumerate(("roll", "pitch"))
        }
        self.plants = {
            "roll": AxisPlant(self.dynamics["roll"], self.contract.roll, timestep),
            "pitch": AxisPlant(self.dynamics["pitch"], self.contract.pitch, timestep),
        }
        # measured=compensate so switching compensation off degrades to plain
        # feedforward rather than silently applying a zero correction.
        self.compensators = {
            name: BacklashCompensator(
                AxisBacklash(lost_motion_rad=backlash if compensate else 0.0,
                             measured=compensate, reference="up"))
            for name in ("roll", "pitch")
        }
        self.reset()

    def reset(self, roll_rad: float = 0.0, pitch_rad: float = 0.0) -> None:
        for name, angle in (("roll", roll_rad), ("pitch", pitch_rad)):
            self.plants[name].reset(angle)
            self.compensators[name].reset()

    @property
    def angles(self) -> tuple[float, float]:
        return (self.plants["roll"].plate_rad, self.plants["pitch"].plate_rad)

    def step(self, roll_rad: float, pitch_rad: float) -> tuple[float, float]:
        """Advance one timestep against a commanded board angle."""
        out = []
        for name, request in (("roll", roll_rad), ("pitch", pitch_rad)):
            aimed = self.compensators[name].target_for(request)
            counts = self.plants[name].axis.angle_to_counts(aimed)
            out.append(self.plants[name].step(counts))
        return (out[0], out[1])

    @property
    def reversals(self) -> tuple[int, int]:
        return (self.compensators["roll"].reversals,
                self.compensators["pitch"].reversals)
