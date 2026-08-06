#!/usr/bin/env python3
"""Command a tilt servo to a position. One shot, no interactive terminal.

Works in raw servo counts because no count->angle calibration exists yet; that
mapping is what sysid produces. 4096 counts = 360 deg, so one count is about
0.088 deg.

By default torque is released when the move finishes, so the board goes limp
rather than being held. Pass ``--hold`` to keep it energised.

Every move is guarded: if the servo's load exceeds ``--max-load`` it stops,
returns to where it started, and releases torque. Servo 1 was measured hitting
load 1044 -- above its own 1000 limit -- roughly 40 counts below its resting
position, so this guard is not theoretical.

Examples:
    python3 tools/servo_move.py --id 2 --delta -40      # 40 counts down
    python3 tools/servo_move.py --id 2 --delta 40       # 40 counts up
    python3 tools/servo_move.py --id 2 --to 2048        # absolute
    python3 tools/servo_move.py --id 2 --degrees -5     # in degrees
    python3 tools/servo_move.py --id 2 --sweep 60       # +-60 and back
    python3 tools/servo_move.py --status                # read, move nothing
    python3 tools/servo_move.py --release               # torque off everything
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.hardware.motion import ramp_to  # noqa: E402
from tag_vision.hardware.sts3215 import (  # noqa: E402
    DEGREES_PER_COUNT,
    Mode,
    Register,
    STS3215Bus,
    counts_to_degrees,
    decode_status,
)

ALL_IDS = (1, 2)


def show(bus: STS3215Bus, servo_id: int, prefix: str = "") -> int:
    state = bus.read_state(servo_id)
    status = decode_status(bus.last_status.get(servo_id, 0))
    torque = bus.read_byte(servo_id, Register.TORQUE_ENABLE)
    print(
        f"{prefix}servo {servo_id}: {state.position:4d} counts "
        f"({counts_to_degrees(state.position):6.2f} deg)  load {state.load:+5d}  "
        f"{state.voltage_v:.1f}V {state.temperature_c}C  torque={torque}"
        + (f"  !{','.join(status)}" if status else "")
    )
    return state.position


def move(bus: STS3215Bus, servo_id: int, target: int, args) -> bool:
    """Ramp to target, aborting on excessive load. True if it arrived."""
    start = bus.read_word(servo_id, Register.PRESENT_POSITION)
    if target != start:
        distance = target - start
        print(
            f"servo {servo_id}: {start} -> {target} "
            f"({distance:+d} counts, {counts_to_degrees(distance):+.2f} deg)"
        )

    result = ramp_to(
        bus,
        servo_id,
        target,
        speed=args.speed,
        accel=args.accel,
        ramp=args.ramp,
        settle=args.settle,
        max_load=args.max_load,
    )

    if not result.arrived:
        print(
            f"  ABORT at {result.aborted_at} counts "
            f"({counts_to_degrees(result.aborted_at - result.start):+.2f} deg "
            f"from start): load {result.abort_load} exceeded {args.max_load}"
        )
        return False

    print(
        f"  arrived {result.final} counts "
        f"(error {result.error:+d}, peak load {result.peak_load})"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--id", type=int, choices=ALL_IDS, default=None,
                        help="servo to move (default: both, for --status/--release)")

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--delta", type=int, help="move this many counts")
    action.add_argument("--to", type=int, help="move to this absolute count")
    action.add_argument("--degrees", type=float, help="move this many degrees")
    action.add_argument("--sweep", type=int,
                        help="sweep +-N counts around the current position")
    action.add_argument("--status", action="store_true", help="read only")
    action.add_argument("--release", action="store_true", help="torque off")

    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--accel", type=int, default=20)
    parser.add_argument("--ramp", type=int, default=10,
                        help="counts per increment while moving (default: 10)")
    parser.add_argument("--settle", type=float, default=0.25,
                        help="seconds to wait per increment (default: 0.25)")
    parser.add_argument("--max-load", type=int, default=350,
                        help="abort above this load (default: 350)")
    parser.add_argument("--hold", action="store_true",
                        help="keep torque on after the move")
    args = parser.parse_args()

    targets = ALL_IDS if args.id is None else (args.id,)

    with STS3215Bus(args.port, args.baud) as bus:
        for servo_id in targets:
            if not bus.ping(servo_id):
                raise SystemExit(
                    f"servo {servo_id} did not answer on {args.port}. "
                    "Check power and run tools/servo_scan.py."
                )

        if args.release:
            for servo_id in targets:
                bus.torque_disable(servo_id)
                show(bus, servo_id, "released  ")
            return

        for servo_id in targets:
            show(bus, servo_id, "before    ")

        if args.status:
            return

        if args.id is None:
            raise SystemExit("--id is required to move a servo")
        servo_id = args.id

        mode = bus.read_byte(servo_id, Register.MODE)
        if mode != Mode.POSITION:
            raise SystemExit(
                f"servo {servo_id} is in {Mode(mode).name} mode; position mode "
                "is required."
            )

        start = bus.read_word(servo_id, Register.PRESENT_POSITION)
        try:
            if args.sweep is not None:
                span = abs(args.sweep)
                print(f"sweeping servo {servo_id} +-{span} counts "
                      f"(+-{span * DEGREES_PER_COUNT:.2f} deg)")
                for target in (start - span, start + span, start):
                    if not move(bus, servo_id, target, args):
                        break
            else:
                if args.delta is not None:
                    target = start + args.delta
                elif args.to is not None:
                    target = args.to
                elif args.degrees is not None:
                    target = start + round(args.degrees / DEGREES_PER_COUNT)
                else:
                    raise SystemExit(
                        "nothing to do: pass --delta, --to, --degrees, --sweep, "
                        "--status, or --release"
                    )
                move(bus, servo_id, target, args)
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            if not args.hold:
                bus.torque_disable(servo_id)
                print(f"servo {servo_id}: torque released")
            else:
                print(f"servo {servo_id}: HOLDING (torque still on)")
            show(bus, servo_id, "after     ")


if __name__ == "__main__":
    main()
