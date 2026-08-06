"""Guarded point-to-point moves, shared by the servo tools.

Every tool that commands motion needs the same two behaviours, and neither is
optional on this rig: step out gradually rather than jumping, and abort if the
servo starts straining. Servo 1 was measured reaching load 1044 -- above its own
1000 limit -- roughly 40 counts below its resting position, so a single
unguarded jump can drive it hard into a mechanical stop.

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
    speed: int = 200,
    accel: int = 20,
    ramp: int = 10,
    settle: float = 0.25,
    max_load: int = 350,
    on_step=None,
) -> MoveResult:
    """Move to ``target`` counts in increments, aborting on excessive load.

    ``on_step`` is called with the intermediate ``ServoState`` after each
    increment, so a caller can display progress without reimplementing the loop.
    """
    start = bus.read_word(servo_id, Register.PRESENT_POSITION)
    target = int(min(max(int(target), 0), COUNTS_PER_REV - 1))

    bus.write_byte(servo_id, Register.ACCELERATION, accel)
    bus.write_word(servo_id, Register.GOAL_SPEED, speed)
    bus.torque_enable(servo_id)

    step = max(1, int(ramp))
    position = start
    peak_load = 0

    while position != target:
        position += max(-step, min(step, target - position))
        bus.set_goal_position(servo_id, position)
        time.sleep(settle)
        state = bus.read_state(servo_id)
        peak_load = max(peak_load, abs(state.load))
        if on_step is not None:
            on_step(state)
        if abs(state.load) > max_load:
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
