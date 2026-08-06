#!/usr/bin/env python3
"""Type a position, the servo goes there. Line-based, works in any terminal.

Unlike ``servo_jog.py`` this needs no raw-mode keyboard, so it works over SSH,
in embedded terminals, and anywhere ``input()`` works.

Positions are raw servo counts: 0-4095 over 360 degrees, about 0.088 deg per
count. Counts rather than board angle because the count->angle mapping is what
sysid measures and does not exist yet.

Commands (servo is remembered between lines):

    2048            move the selected servo to count 2048
    1 2048          move servo 1 to count 2048
    +30  /  -30     relative move
    d 5  /  d -5    relative move in degrees
    id 2            select servo 2
    s               status of both servos
    home            back to where each servo was at startup
    on              torque ON, hold current position (also: torque, hold)
    on both         torque ON for both servos
    off             torque OFF both servos (also: free, release)
    step 20         set the ramp increment
    load 500        set the load abort threshold
    q               quit (releases torque)

Example session:

    servo 2 > 2048
    servo 2 > +40
    servo 2 > d -2.5
    servo 2 > q
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.hardware.motion import ramp_to  # noqa: E402
from tag_vision.hardware.sts3215 import (  # noqa: E402
    COUNTS_PER_REV,
    DEGREES_PER_COUNT,
    Mode,
    Register,
    STS3215Bus,
    STSError,
    counts_to_degrees,
    decode_status,
)

ALL_IDS = (1, 2)


class Console:
    def __init__(self, bus: STS3215Bus, args):
        self.bus = bus
        self.args = args
        self.servo_id = args.id
        self.hold = args.hold
        self.step = args.ramp
        self.max_load = args.max_load
        self.home = {
            servo_id: bus.read_word(servo_id, Register.PRESENT_POSITION)
            for servo_id in ALL_IDS
        }

    # ---- reporting ---------------------------------------------------------
    def status(self) -> None:
        for servo_id in ALL_IDS:
            state = self.bus.read_state(servo_id)
            flags = decode_status(self.bus.last_status.get(servo_id, 0))
            torque = self.bus.read_byte(servo_id, Register.TORQUE_ENABLE)
            marker = "*" if servo_id == self.servo_id else " "
            print(
                f" {marker}servo {servo_id}: {state.position:4d} counts "
                f"({counts_to_degrees(state.position):6.2f} deg)  "
                f"home {self.home[servo_id]:4d} "
                f"({state.position - self.home[servo_id]:+d})  "
                f"load {state.load:+5d}  {state.voltage_v:.1f}V "
                f"{state.temperature_c}C  torque={torque}"
                + (f"  !{','.join(flags)}" if flags else "")
            )

    # ---- motion ------------------------------------------------------------
    def goto(self, target: int) -> None:
        target = int(round(target))
        if not 0 <= target < COUNTS_PER_REV:
            print(f"  {target} is outside 0..{COUNTS_PER_REV - 1}")
            return

        result = ramp_to(
            self.bus,
            self.servo_id,
            target,
            speed=self.args.speed,
            accel=self.args.accel,
            ramp=self.step,
            settle=self.args.settle,
            max_load=self.max_load,
        )

        if result.arrived:
            print(
                f"  {result.start} -> {result.final} "
                f"({result.final - result.start:+d} counts, "
                f"{counts_to_degrees(result.final - result.start):+.2f} deg), "
                f"error {result.error:+d}, peak load {result.peak_load}"
            )
            if not self.hold:
                self.bus.torque_disable(self.servo_id)
        else:
            print(
                f"  ABORTED at {result.aborted_at} "
                f"(load {result.abort_load} > {self.max_load}); "
                f"returned to {result.final}, torque off"
            )

    def go_home(self) -> None:
        for servo_id in ALL_IDS:
            self.servo_id, previous = servo_id, self.servo_id
            print(f" servo {servo_id} -> home {self.home[servo_id]}")
            self.goto(self.home[servo_id])
            self.servo_id = previous

    def torque_on(self, both: bool = False) -> None:
        """Energise and hold the current position, without moving.

        The goal register is set to where the servo actually is before torque is
        enabled. Skipping that would make the servo snap to whatever stale goal
        was left over from a previous move.
        """
        targets = ALL_IDS if both else (self.servo_id,)
        for servo_id in targets:
            position = self.bus.read_word(servo_id, Register.PRESENT_POSITION)
            self.bus.write_byte(servo_id, Register.ACCELERATION, self.args.accel)
            self.bus.write_word(servo_id, Register.GOAL_SPEED, self.args.speed)
            self.bus.set_goal_position(servo_id, position)
            self.bus.torque_enable(servo_id)
            print(f"  servo {servo_id}: torque ON, holding {position}")
        # Torque stays on after subsequent moves too, which is what someone
        # asking for torque almost always wants.
        self.hold = True

    def free(self) -> None:
        for servo_id in ALL_IDS:
            self.bus.torque_disable(servo_id)
        self.hold = False
        print("  torque released on both servos")

    # ---- command parsing ---------------------------------------------------
    def handle(self, line: str) -> bool:
        """Return False to quit."""
        parts = line.split()
        if not parts:
            return True
        head = parts[0].lower()

        if head in ("q", "quit", "exit"):
            return False
        if head in ("s", "status"):
            self.status()
            return True
        if head == "home":
            self.go_home()
            return True
        if head in ("free", "release", "off"):
            self.free()
            return True
        if head in ("on", "torque", "hold"):
            self.torque_on(both=len(parts) > 1 and parts[1].lower() == "both")
            return True
        if head == "id" and len(parts) == 2:
            self._select(parts[1])
            return True
        if head == "step" and len(parts) == 2:
            self.step = max(1, int(parts[1]))
            print(f"  ramp increment {self.step} counts")
            return True
        if head == "load" and len(parts) == 2:
            self.max_load = max(1, int(parts[1]))
            print(f"  load abort threshold {self.max_load}")
            return True
        if head == "d" and len(parts) == 2:
            degrees = float(parts[1])
            current = self.bus.read_word(self.servo_id, Register.PRESENT_POSITION)
            self.goto(current + degrees / DEGREES_PER_COUNT)
            return True

        # "1 2048" -- servo id then target
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in ALL_IDS:
            self._select(parts[0])
            self.goto(int(parts[1]))
            return True

        # "+30", "-30", or a bare absolute position
        token = parts[0]
        try:
            if token[0] in "+-":
                current = self.bus.read_word(self.servo_id, Register.PRESENT_POSITION)
                self.goto(current + int(token))
            else:
                self.goto(int(token))
        except ValueError:
            print(f"  don't understand {line!r} -- try a number, +N, -N, d N, s, q")
        return True

    def _select(self, token: str) -> None:
        try:
            servo_id = int(token)
        except ValueError:
            print(f"  bad servo id {token!r}")
            return
        if servo_id not in ALL_IDS:
            print(f"  servo id must be one of {ALL_IDS}")
            return
        self.servo_id = servo_id
        print(f"  selected servo {servo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--id", type=int, choices=ALL_IDS, default=2,
                        help="servo selected at startup (default: 2)")
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--accel", type=int, default=20)
    parser.add_argument("--ramp", type=int, default=10,
                        help="counts per increment while moving (default: 10)")
    parser.add_argument("--settle", type=float, default=0.25)
    parser.add_argument("--max-load", type=int, default=350)
    parser.add_argument("--hold", action="store_true",
                        help="keep torque on after each move")
    args = parser.parse_args()

    with STS3215Bus(args.port, args.baud) as bus:
        for servo_id in ALL_IDS:
            if not bus.ping(servo_id):
                raise SystemExit(
                    f"servo {servo_id} did not answer on {args.port}. "
                    "Check power and run tools/servo_scan.py."
                )
            mode = bus.read_byte(servo_id, Register.MODE)
            if mode != Mode.POSITION:
                raise SystemExit(
                    f"servo {servo_id} is in {Mode(mode).name} mode; "
                    "position mode is required."
                )

        console = Console(bus, args)
        print(__doc__.split("Commands")[1].split("Example session")[0].strip())
        print()
        console.status()
        print()

        try:
            while True:
                try:
                    line = input(f"servo {console.servo_id} > ")
                except EOFError:
                    break
                try:
                    if not console.handle(line):
                        break
                except STSError as exc:
                    print(f"  bus error: {exc}")
                except ValueError as exc:
                    print(f"  {exc}")
        except KeyboardInterrupt:
            print()
        finally:
            for servo_id in ALL_IDS:
                try:
                    bus.torque_disable(servo_id)
                except STSError:
                    pass
            print("Torque released on both servos.")


if __name__ == "__main__":
    main()
