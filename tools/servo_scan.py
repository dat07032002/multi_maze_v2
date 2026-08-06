#!/usr/bin/env python3
"""Discover STS3215 servos on the bus. Strictly read-only.

Scans baud rates and IDs with PING, then dumps each servo's configuration. No
write is ever issued, so this is safe to run against hardware whose state is
unknown -- which is the point: the register map and byte order have to be
confirmed against a real servo before anything commands motion.

Examples:
    python3 tools/servo_scan.py
    python3 tools/servo_scan.py --port /dev/ttyUSB0 --baud 1000000
    python3 tools/servo_scan.py --watch 2          # live position readout
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.hardware.sts3215 import (  # noqa: E402
    BAUDRATE_TABLE,
    STS3215Bus,
    STSError,
    Mode,
    counts_to_degrees,
    decode_status,
)


def scan_baudrates(port: str, id_range: range, timeout_s: float):
    """Try each documented baud rate until one answers."""
    for index in sorted(BAUDRATE_TABLE):
        baudrate = BAUDRATE_TABLE[index]
        try:
            with STS3215Bus(port, baudrate, timeout_s) as bus:
                found = bus.scan(id_range)
        except (OSError, STSError) as exc:
            print(f"  {baudrate:>9} bps  port error: {exc}")
            continue
        marker = "<-- servos here" if found else ""
        print(f"  {baudrate:>9} bps  ids {found if found else '[]'} {marker}")
        if found:
            return baudrate, found
    return None, []


def describe(config: dict) -> str:
    mode = config["mode"]
    try:
        mode_name = Mode(mode).name
    except ValueError:
        mode_name = f"UNKNOWN({mode})"
    return (
        f"  mode              {mode_name}\n"
        f"  model             {config['model']}\n"
        f"  firmware          {config['firmware'][0]}.{config['firmware'][1]}\n"
        f"  PID               P={config['p_coefficient']} "
        f"D={config['d_coefficient']} I={config['i_coefficient']}\n"
        f"  acceleration      {config['acceleration']}\n"
        f"  goal speed        {config['goal_speed']}\n"
        f"  angle limits      {config['min_angle_limit']}..{config['max_angle_limit']}"
        f"  ({counts_to_degrees(config['min_angle_limit']):.1f}"
        f"..{counts_to_degrees(config['max_angle_limit']):.1f} deg)\n"
        f"  dead zone         cw={config['cw_dead_zone']} ccw={config['ccw_dead_zone']}\n"
        f"  min startup force {config['min_startup_force']}\n"
        f"  torque limit      {config['torque_limit']} / max "
        f"{config['max_torque_limit']}\n"
        f"  torque enabled    {bool(config['torque_enable'])}\n"
        f"  position offset   {config['position_offset']}\n"
        f"  phase             {config['phase']}\n"
        f"  response level    {config['response_level']}\n"
        f"  voltage           {config['present_voltage_v']:.1f} V"
        f"  (limits {config['min_voltage_v']:.1f}..{config['max_voltage_v']:.1f} V)\n"
        f"  temperature       {config['present_temperature_c']} C\n"
        f"  status            {_status_text(config['packet_status'])}"
    )


def _status_text(status: int) -> str:
    flags = decode_status(status)
    if not flags:
        return "ok"
    return f"0x{status:02x} -- " + ", ".join(flags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=None,
                        help="skip the baud scan and use this rate")
    parser.add_argument("--max-id", type=int, default=20,
                        help="highest servo ID to probe (default: 20)")
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--watch", type=float, default=0.0,
                        help="after scanning, stream positions for N seconds")
    parser.add_argument("--save", default=None,
                        help="write the discovered configuration to this JSON file")
    args = parser.parse_args()

    id_range = range(0, args.max_id + 1)

    if args.baud is None:
        print(f"Scanning {args.port} for servos (read-only, PING only)...")
        baudrate, found = scan_baudrates(args.port, id_range, args.timeout)
        if not found:
            raise SystemExit(
                "\nNo servos answered at any baud rate.\n"
                "Check: bus power on, data line wiring, and that the adapter is "
                f"really {args.port} (see: ls /dev/ttyUSB*)."
            )
    else:
        baudrate = args.baud
        with STS3215Bus(args.port, baudrate, args.timeout) as bus:
            found = bus.scan(id_range)
        if not found:
            raise SystemExit(f"No servos answered at {baudrate} bps.")

    print(f"\nFound {len(found)} servo(s) at {baudrate} bps: {found}\n")

    configs = {}
    with STS3215Bus(args.port, baudrate, args.timeout) as bus:
        for servo_id in found:
            config = bus.read_config(servo_id)
            configs[servo_id] = config
            print(f"servo {servo_id}")
            print(describe(config))
            state = bus.read_state(servo_id)
            print(
                f"  position          {state.position} counts "
                f"({counts_to_degrees(state.position):.2f} deg)\n"
                f"  speed / load      {state.speed} / {state.load}\n"
            )

        if args.watch > 0:
            print(
                "Back-drive each servo by hand and confirm the reading tracks it.\n"
                "If position jumps erratically or never changes, the byte order or "
                "register address is wrong.\n"
            )
            deadline = time.monotonic() + args.watch
            while time.monotonic() < deadline:
                parts = []
                for servo_id in found:
                    state = bus.read_state(servo_id)
                    parts.append(
                        f"id{servo_id}: {state.position:4d} "
                        f"({counts_to_degrees(state.position):6.2f} deg) "
                        f"v={state.speed:5d} load={state.load:5d}"
                    )
                sys.stdout.write("\r" + " | ".join(parts) + "   ")
                sys.stdout.flush()
                time.sleep(0.05)
            print()

    if args.save:
        payload = {
            "port": args.port,
            "baudrate": baudrate,
            "servo_ids": found,
            "configs": {str(k): v for k, v in configs.items()},
        }
        Path(args.save).write_text(json.dumps(payload, indent=2) + "\n",
                                   encoding="utf-8")
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
