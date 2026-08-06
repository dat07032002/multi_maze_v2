#!/usr/bin/env python3
"""Bring-up viewer for the BNO086 tilt stream, and the tool that captures zero.

This proves the firmware and the host reader before any servo is commanded. Run
it first; nothing downstream is trustworthy until the gate below passes.

    Gate: stable alpha/beta with static noise well under 0.05 deg, and no
    dropped frames at 200 Hz.

Capturing the level zero is the other job. The board is currently level with the
IMU mounted on it, so the reference is captured in place rather than derived --
which is why it must be captured before anything disturbs the plate.

Examples:
    python3 tools/imu_monitor.py                     # live readout, Ctrl-C quits
    python3 tools/imu_monitor.py --seconds 20        # fixed-length noise check
    python3 tools/imu_monitor.py --capture-zero      # write calib/imu_zero.json
    python3 tools/imu_monitor.py --capture-zero --check-torque-shift
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tag_vision.core.board_pose import angles_from_rotation  # noqa: E402
from tag_vision.hardware.imu import DEFAULT_PORT, BNO086Stream, ImuError  # noqa: E402
from tag_vision.hardware.sts3215 import Register, STS3215Bus  # noqa: E402

DEFAULT_ZERO_PATH = Path(__file__).resolve().parent.parent / "calib" / "imu_zero.json"
SERVO_IDS = (1, 2)

# The Step 2 gate. Static noise above this means the sensor or the mounting is
# not good enough to measure a servo against.
NOISE_GATE_DEG = 0.05


def read_servo_counts(port: str | None) -> dict[int, int]:
    """Present position of both tilt servos, or an empty dict if unreachable.

    Recorded alongside the zero because the zero is only meaningful together
    with the servo positions that produced it: that pair is the center_counts
    candidate the contract needs.
    """
    try:
        with STS3215Bus(port=port) as bus:
            return {i: bus.read_word(i, Register.PRESENT_POSITION)
                    for i in SERVO_IDS}
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        print(f"  warning: could not read servo positions: {exc}")
        return {}


def set_torque(port: str, enabled: bool) -> bool:
    try:
        with STS3215Bus(port=port) as bus:
            for servo_id in SERVO_IDS:
                if enabled:
                    bus.torque_enable(servo_id)
                else:
                    bus.torque_disable(servo_id)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: could not set torque: {exc}")
        return False


def monitor(imu: BNO086Stream, seconds: float | None) -> int:
    """Live readout. Returns a process exit code based on the Step 2 gate."""
    print(f"reading {imu.port}; Ctrl-C to stop")
    print(f"{'alpha deg':>10} {'beta deg':>10} {'Hz':>7} {'drop':>6} "
          f"{'crc':>5} {'resync':>7} {'acc':>4}")

    alphas: list[float] = []
    betas: list[float] = []
    start = time.monotonic()
    last_print = 0.0
    count = 0
    first_micros: int | None = None
    last_micros: int | None = None

    try:
        while seconds is None or time.monotonic() - start < seconds:
            sample = imu.read_sample(timeout=1.0)
            if sample is None:
                for message in imu.status_messages[-3:]:
                    print(f"  firmware: {message}")
                print("  no samples; is the firmware flashed and running?")
                time.sleep(0.5)
                continue

            alpha, beta = imu.angles(sample)
            alphas.append(math.degrees(alpha))
            betas.append(math.degrees(beta))
            count += 1
            if first_micros is None:
                first_micros = sample.esp_micros
            last_micros = sample.esp_micros

            now = time.monotonic()
            if now - last_print >= 0.2:
                last_print = now
                elapsed = now - start
                rate = count / elapsed if elapsed > 0 else 0.0
                print(f"\r{alphas[-1]:10.4f} {betas[-1]:10.4f} {rate:7.1f} "
                      f"{imu.dropped:6d} {imu.crc_errors:5d} "
                      f"{imu.resyncs:7d} {sample.accuracy:4d}",
                      end="", flush=True)
    except KeyboardInterrupt:
        pass

    print()
    if count < 2:
        print("no usable samples")
        return 1

    # Device-clock rate is the honest one: the host rate also measures how often
    # this loop got scheduled.
    span_s = (last_micros - first_micros) * 1e-6 if last_micros else 0.0
    device_rate = (count - 1) / span_s if span_s > 0 else float("nan")

    alpha_sd = statistics.pstdev(alphas)
    beta_sd = statistics.pstdev(betas)

    print(f"\nsamples          {count}")
    print(f"device rate      {device_rate:.1f} Hz")
    print(f"alpha            mean {statistics.fmean(alphas):+.4f} deg  "
          f"sd {alpha_sd:.4f} deg")
    print(f"beta             mean {statistics.fmean(betas):+.4f} deg  "
          f"sd {beta_sd:.4f} deg")
    print(f"dropped {imu.dropped}  crc errors {imu.crc_errors}  "
          f"resyncs {imu.resyncs}")

    try:
        offset, rtt = imu.estimate_clock_offset(samples=11)
        print(f"link round-trip  {rtt * 1e3:.2f} ms (best of 11)")
        print(f"clock offset     {offset:.6f} s")
    except ImuError as exc:
        print(f"clock offset     unavailable: {exc}")

    ok = True
    if max(alpha_sd, beta_sd) > NOISE_GATE_DEG:
        print(f"\nGATE FAILED: static noise exceeds {NOISE_GATE_DEG} deg")
        ok = False
    if imu.dropped:
        print(f"\nGATE FAILED: {imu.dropped} dropped frames")
        ok = False
    if ok:
        print("\nGATE PASSED")
    return 0 if ok else 1


def capture_zero(imu: BNO086Stream, args) -> int:
    servo_port = args.servo_port

    if args.check_torque_shift:
        # The board may be level only because it is hanging limp. If energising
        # the servos moves it, the resting pose is not the pose the control loop
        # will ever see, and zeroing against it would bias every angle.
        print("capturing with torque OFF ...")
        set_torque(servo_port, False)
        time.sleep(0.5)
        imu.drain()
        limp_rotation, limp_n = imu.capture_zero(seconds=args.seconds)
        limp_alpha, limp_beta = angles_from_rotation(limp_rotation)
        print(f"  torque off: alpha {math.degrees(limp_alpha):+.4f} "
              f"beta {math.degrees(limp_beta):+.4f} deg  ({limp_n} samples)")

        print("enabling torque ...")
        if not set_torque(servo_port, True):
            print("  aborting: cannot compare poses without torque control")
            return 1
        time.sleep(1.0)

    print(f"capturing zero over {args.seconds:.1f} s ...")
    imu.drain()
    # Clear any previous zero so the capture is absolute, not relative to an
    # older reference that may itself be stale.
    imu.zero_rotation = None
    rotation, n = imu.capture_zero(seconds=args.seconds)

    if args.check_torque_shift:
        shift_alpha, shift_beta = angles_from_rotation(
            limp_rotation.T @ rotation)
        print(f"  torque on minus torque off: "
              f"alpha {math.degrees(shift_alpha):+.4f} "
              f"beta {math.degrees(shift_beta):+.4f} deg")
        if max(abs(shift_alpha), abs(shift_beta)) > math.radians(0.1):
            print("  NOTE: the board moved when torque was applied. The powered "
                  "pose is the correct zero, and this one is it.")

    counts = read_servo_counts(servo_port)
    print(f"  servo counts: {counts}")
    if len(counts) != len(SERVO_IDS):
        print("  WARNING: zero saved without both servo positions. The zero is "
              "only meaningful paired with the counts that produced it.")

    imu.save_zero(args.output, extra={
        "samples_averaged": n,
        "servo_counts_at_zero": {str(k): v for k, v in counts.items()},
        "torque_enabled": bool(args.check_torque_shift),
        "capture_seconds": args.seconds,
    })
    print(f"\nwrote {args.output}")
    print("Do not disturb the plate or re-mount the IMU: this reference is "
          "captured in place and cannot be recovered by eye.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=None,
                        help="ESP32 serial port; auto-resolved by USB id")
    parser.add_argument("--servo-port", default=None,
                        help="FEETECH bus port; auto-resolved by USB id")
    parser.add_argument("--seconds", type=float, default=None,
                        help="run for this long, then report; default is until Ctrl-C")
    parser.add_argument("--capture-zero", action="store_true",
                        help="capture the level reference and write it to disk")
    parser.add_argument("--check-torque-shift", action="store_true",
                        help="capture torque-off and torque-on, and report the difference")
    parser.add_argument("--output", type=Path, default=DEFAULT_ZERO_PATH,
                        help="zero file to write (default: %(default)s)")
    args = parser.parse_args()

    # Only meaningful when both were given explicitly; resolution keeps them
    # apart otherwise. This guard once passed while --servo-port pointed at the
    # IMU, because the numbers had swapped underneath a hardcoded default.
    if args.port is not None and args.port == args.servo_port:
        parser.error("--port and --servo-port must differ; the IMU is on the "
                     "CP2102 and the servos are on the CH340")

    try:
        imu = BNO086Stream(port=args.port)
    except Exception as exc:  # noqa: BLE001
        print(f"could not open {args.port}: {exc}")
        return 1

    with imu:
        if args.capture_zero:
            if args.seconds is None:
                args.seconds = 3.0
            return capture_zero(imu, args)
        return monitor(imu, args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
