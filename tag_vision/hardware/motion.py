"""Guarded point-to-point moves, shared by the servo tools.

Every tool that commands motion needs the same two behaviours, and neither is
optional on this rig: step out gradually rather than jumping, and abort if the
servo stalls.

"Stalls" is doing real work in that sentence. An earlier version aborted on load
alone and rejected every upward move, which read convincingly as a seized
mechanism. The immediate cause was a decoding bug -- PRESENT_LOAD's direction
lives in bit 10, and reading it as bit 15 made loads in one direction report as
1024 plus their true value, so a gentle -24 arrived as +1048. But the guard was
wrong independently of that: load alone does not distinguish a servo working
from a servo stuck, and the encoder kept advancing throughout.

The test is therefore high load *with no progress*, held over several samples.

Backing off to the start before releasing torque matters: cutting torque while
wedged leaves the servo resting against the stop, whereas retreating first lets
it relax.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .sts3215 import COUNTS_PER_REV, Register, STS3215Bus


@dataclass
class MoveResult:
    arrived: bool
    start: int
    target: int
    final: int
    peak_load: int
    aborted_at: int | None = None
    abort_load: int | None = None

    @property
    def error(self) -> int:
        return self.final - self.target


def ramp_to(
    bus: STS3215Bus,
    servo_id: int,
    target: int,
    *,
    speed: int | None = None,
    accel: int | None = None,
    ramp: int = 10,
    settle: float = 0.25,
    max_load: int = 350,
    stall_counts: int = 2,
    stall_checks: int = 3,
    motion_epsilon: float = 0.01,
    on_step=None,
    motion_probe=None,
) -> MoveResult:
    """Move to ``target`` counts in increments, aborting on a stall.

    A stall is high load *and* no progress, sustained over ``stall_checks``
    consecutive samples. Load on its own does not abort: on this rig a servo
    lifting the plate reads around 1080 while travelling perfectly well,
    against about 72 lowering it, so a load-only test refuses every upward move.

    ``motion_probe`` is an optional callable returning a scalar measurement of
    the thing you actually care about moving -- board angle in degrees, say.
    When given, progress is judged from it rather than from the encoder, with
    ``motion_epsilon`` as the smallest change that counts. Prefer it: the
    encoder reports where the servo put its own shaft, which on this rig has
    already differed sharply from where the plate went.

    ``on_step`` is called with the intermediate ``ServoState`` after each
    increment, so a caller can display progress without reimplementing the loop.
    """
    start = bus.read_word(servo_id, Register.PRESENT_POSITION)
    target = int(min(max(int(target), 0), COUNTS_PER_REV - 1))

    # Only write these when explicitly asked. They set the servo's motion
    # profile, and the measured latency and rise time are properties of them:
    # rewriting them here on every call meant any caller could invalidate the
    # calibration silently. The canonical values live in
    # ``STS3215Bus.CANONICAL_CONFIG`` and are applied once, at startup.
    if accel is not None:
        bus.write_byte(servo_id, Register.ACCELERATION, accel)
    if speed is not None:
        if speed == 0:
            raise ValueError(
                "GOAL_SPEED 0 is not 'maximum' on this firmware: a 160-count "
                "step under it moved the board 0.031 deg in 1.26 s. Pass a "
                "positive speed, or None to inherit the configured value.")
        bus.write_word(servo_id, Register.GOAL_SPEED, speed)
    bus.torque_enable(servo_id)

    step = max(1, int(ramp))
    position = start
    peak_load = 0
    last_actual = start
    last_observed = None
    stalled_for = 0

    while position != target:
        position += max(-step, min(step, target - position))
        bus.set_goal_position(servo_id, position)
        time.sleep(settle)
        state = bus.read_state(servo_id)
        peak_load = max(peak_load, abs(state.load))
        if on_step is not None:
            on_step(state)

        # High load alone is not a fault. Lifting a heavy plate against gravity
        # legitimately draws far more than lowering it: measured on this rig,
        # the same servo reads ~1080 while raising and ~72 while lowering, and
        # in both cases the encoder is advancing normally. Aborting on load
        # alone rejected every upward move and looked convincingly like a
        # seized mechanism.
        #
        # A real jam is high load *with no progress*, so both must hold, and for
        # several consecutive checks -- a single sample can land mid-transient
        # while the servo is still breaking away.
        # Progress means the *board* moved, not the shaft, whenever a probe can
        # tell us. The encoder only reports where the servo put itself, and this
        # rig has already shown the two coming apart: servo 1 advanced 200
        # counts through its dead zone while the plate moved 0.08 deg. A guard
        # watching the encoder alone would call that healthy travel.
        if motion_probe is not None:
            observed = motion_probe()
            progressed = (last_observed is None
                          or abs(observed - last_observed) >= motion_epsilon)
            last_observed = observed
        else:
            progressed = abs(state.position - last_actual) >= stall_counts
        last_actual = state.position
        if abs(state.load) > max_load and not progressed:
            stalled_for += 1
        else:
            stalled_for = 0

        if stalled_for >= stall_checks:
            bus.set_goal_position(servo_id, start)
            time.sleep(0.5)
            bus.torque_disable(servo_id)
            final = bus.read_word(servo_id, Register.PRESENT_POSITION)
            return MoveResult(
                arrived=False,
                start=start,
                target=target,
                final=final,
                peak_load=peak_load,
                aborted_at=state.position,
                abort_load=state.load,
            )

    time.sleep(0.2)
    final_state = bus.read_state(servo_id)
    peak_load = max(peak_load, abs(final_state.load))
    return MoveResult(
        arrived=True,
        start=start,
        target=target,
        final=final_state.position,
        peak_load=peak_load,
    )
