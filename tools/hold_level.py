#!/usr/bin/env python3
"""Move the board to a level position and hold it there.

Every other servo tool in this repo releases torque when it finishes, so the
board goes limp and sags back. That is the right default for jogging and for
sysid, and the wrong one here: this tool exists to put the board somewhere and
keep it there.

So torque goes on at startup and **stays on when you quit**. Nothing releases it
except the ``off`` command or ``--release``. Quitting deliberately leaves the
board held.

Torque is enabled at the position the servos are already in, so starting the
tool never moves anything.

Commands:

    -20  /  +20     move the selected servo by that many counts
    3986            move the selected servo to that absolute count
    id 2            select servo 2 (also: ``2 -20`` moves servo 2 directly)
    s               status of both servos
    save            record the current position as the level pose
    off             release torque (board goes limp)
    q               quit, LEAVING THE BOARD HELD

Examples:
    python3 tools/hold_level.py
    python3 tools/hold_level.py --set 1=3986 --set 2=2930
    python3 tools/hold_level.py --release
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.hardware.motion import ramp_to  # noqa: E402
from tag_vision.hardware.sts3215 import (  # noqa: E402
    COUNTS_PER_REV,
    DEGREES_PER_COUNT,
    Register,
    STS3215Bus,
)

ROOT = Path(__file__).resolve().parents[1]
LEVEL_FILE = ROOT / "calib" / "board_level_counts.json"
SERVO_IDS = (1, 2)


def show(bus: STS3215Bus) -> None:
    for servo_id in SERVO_IDS:
        state = bus.read_state(servo_id)
        torque = bus.read_byte(servo_id, Register.TORQUE_ENABLE)
        headroom = min(state.position, COUNTS_PER_REV - 1 - state.position)
        warn = "  <- near end of travel" if headroom < 100 else ""
        print(f"  servo {servo_id}: {state.position:4d} counts  "
              f"load {state.load:+5d}  {state.voltage_v:.1f}V  "
              f"{state.temperature_c}C  torque={'on' if torque else 'off'}"
              f"{warn}")


def move(bus: STS3215Bus, servo_id: int, target: int, args) -> bool:
    """Move and stay energised. Returns True if it arrived."""
    target = int(min(max(target, 0), COUNTS_PER_REV - 1))
    start = bus.read_word(servo_id, Register.PRESENT_POSITION)

    result = ramp_to(bus, servo_id, target, max_load=args.max_load,
                     ramp=args.step, settle=args.settle)

    if not result.arrived:
        # ramp_to backs off and cuts torque on an abort, which is correct when
        # something is straining. Re-energise where it ended up so the board is
        # still held rather than dropping limp.
        print(f"  ABORTED at {result.aborted_at} "
              f"(load {result.abort_load} > {args.max_load}); "
              f"returned to {result.final}")
        bus.torque_enable(servo_id)
        print("  re-energised at the backed-off position; board still held")
        return False

    delta = result.final - start
    print(f"  servo {servo_id}: {start} -> {result.final} "
          f"({delta:+d} counts, {delta * DEGREES_PER_COUNT:+.2f} deg), "
          f"peak load {result.peak_load}  [holding]")
    return True


def save_level(bus: STS3215Bus) -> None:
    counts = {str(i): bus.read_word(i, Register.PRESENT_POSITION)
              for i in SERVO_IDS}
    LEVEL_FILE.write_text(json.dumps({
        "servo_counts": counts,
        "captured": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Servo positions at which the board was judged level. These are "
            "raw counts: no count->angle calibration exists yet, so this "
            "records where level is, not what level means in radians. Pair it "
            "with calib/imu_zero.json, which records the same pose as a "
            "rotation."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  saved {counts} to {LEVEL_FILE}")


def run(bus: STS3215Bus, args) -> int:
    print("current position:")
    show(bus)

    # Hold where they already are: enabling torque against the present position
    # means startup never moves the board.
    for servo_id in SERVO_IDS:
        bus.set_goal_position(
            servo_id, bus.read_word(servo_id, Register.PRESENT_POSITION))
        bus.torque_enable(servo_id)
    print("\ntorque ON, holding current position")
    print("type -20 / +20 / 3986 / id 2 / s / save / off / q "
          "(q keeps the board held)\n")

    selected = args.id
    while True:
        try:
            line = input(f"servo {selected} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        parts = line.split()
        command = parts[0].lower()

        if command in ("q", "quit", "exit"):
            break
        if command in ("s", "status"):
            show(bus)
            continue
        if command == "save":
            save_level(bus)
            continue
        if command in ("off", "release", "free"):
            for servo_id in SERVO_IDS:
                bus.torque_disable(servo_id)
            print("  torque OFF, board is limp")
            continue
        if command == "id" and len(parts) > 1:
            try:
                candidate = int(parts[1])
            except ValueError:
                print("  usage: id 1")
                continue
            if candidate not in SERVO_IDS:
                print(f"  servo must be one of {SERVO_IDS}")
                continue
            selected = candidate
            continue

        # "2 -20" targets servo 2 explicitly; a bare value uses the selection.
        target_id = selected
        value_text = parts[0]
        if len(parts) > 1 and parts[0] in ("1", "2"):
            target_id = int(parts[0])
            value_text = parts[1]

        try:
            value = int(value_text)
        except ValueError:
            print(f"  did not understand {line!r}")
            continue

        current = bus.read_word(target_id, Register.PRESENT_POSITION)
        # A leading sign means relative; a bare number is absolute.
        relative = value_text[0] in "+-"
        target = current + value if relative else value
        move(bus, target_id, target, args)

    print("\nquitting with the board HELD (torque still on).")
    print("Release it with:  python3 tools/hold_level.py --release")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--id", type=int, default=1, choices=SERVO_IDS,
                        help="servo selected at startup")
    parser.add_argument("--set", action="append", default=[], metavar="ID=COUNT",
                        help="move a servo and hold, without the prompt")
    parser.add_argument("--release", action="store_true",
                        help="release torque on both servos and exit")
    parser.add_argument("--max-load", type=int, default=350)
    parser.add_argument("--step", type=int, default=10,
                        help="ramp increment in counts")
    parser.add_argument("--settle", type=float, default=0.15)
    args = parser.parse_args()

    with STS3215Bus(port=args.port) as bus:
        if args.release:
            for servo_id in SERVO_IDS:
                bus.torque_disable(servo_id)
            print("torque OFF on both servos; the board is limp")
            show(bus)
            return 0

        for servo_id in SERVO_IDS:
            state = bus.read_state(servo_id)
            if not 6.0 <= state.voltage_v <= 8.4:
                print(f"ABORT: servo {servo_id} at {state.voltage_v:.1f} V, "
                      "outside 6.0-8.4 V")
                return 1

        if args.set:
            for assignment in args.set:
                if "=" not in assignment:
                    print(f"  --set needs ID=COUNT, got {assignment!r}")
                    return 1
                sid_text, count_text = assignment.split("=", 1)
                try:
                    servo_id, count = int(sid_text), int(count_text)
                except ValueError:
                    print(f"  --set needs integers, got {assignment!r}")
                    return 1
                if servo_id not in SERVO_IDS:
                    print(f"  servo must be one of {SERVO_IDS}")
                    return 1
                bus.torque_enable(servo_id)
                move(bus, servo_id, count, args)
            print("\nboard HELD (torque still on). "
                  "Release with --release when done.")
            return 0

        return run(bus, args)


if __name__ == "__main__":
    raise SystemExit(main())
