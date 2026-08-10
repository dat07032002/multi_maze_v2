#!/usr/bin/env python3
"""Does feedforward backlash compensation actually help on the board?

Run after any sysid change that touches the calibration or the lost motion.

Commands the same angles from alternating directions and measures where the
plate really lands, with compensation off and then on. The figure of merit is
the spread at a repeated target: without compensation, approaching 0 deg from
above and from below should differ by roughly the lost motion; with it, they
should agree.

This is the test that distinguishes a correct backlash inverse from one applied
with the wrong sign, which would double the spread rather than remove it.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.servo_contract import ServoContract
from tag_vision.control.tilt import AxisBacklash, TiltController, TiltLimits
from tag_vision.hardware.imu import BNO086Stream
from tag_vision.hardware.sts3215 import STS3215Bus

ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_DEG = [0, +2, 0, -2, 0, +2, 0, -2, 0]
SETTLE_S = 1.4

latest = {"a": 0.0, "b": 0.0}
stop = threading.Event()


def reader(imu):
    while not stop.is_set():
        s = imu.read_sample(timeout=0.5)
        if s:
            a, b = imu.angles(s)
            latest["a"], latest["b"] = a, b


def settled(seconds=0.7):
    t0 = time.monotonic()
    a, b = [], []
    while time.monotonic() - t0 < seconds:
        a.append(latest["a"]); b.append(latest["b"])
        time.sleep(0.005)
    return math.degrees(statistics.fmean(a)), math.degrees(statistics.fmean(b))


def interior_median(gaps):
    """Lost motion from the flat part of the hysteresis loop.

    The gap falls to zero at both turnarounds because that is where the two
    legs meet, so the endpoints describe the sweep rather than the mechanism.
    """
    interior = sorted(abs(g) for g in gaps[1:-1])
    return statistics.median(interior)


def measure_backlash(run_path):
    d = json.load(open(run_path))
    out = {}
    for sid, f in d["experiment_AB"].items():
        axis = {"alpha": "roll", "beta": "pitch"}[f["angle"]]
        key = f["angle"]
        up = {p["commanded_counts"]: p[f"{key}_rad"] for p in f["points"]
              if p["direction"] == "up"}
        dn = {p["commanded_counts"]: p[f"{key}_rad"] for p in f["points"]
              if p["direction"] == "down"}
        gaps = [dn[c] - up[c] for c in sorted(set(up) & set(dn))]
        out[axis] = interior_median(gaps)
    return out


def run_sequence(ctl, label):
    print(f"\n== {label} ==")
    print(f"{'target':>8}{'roll':>10}{'pitch':>10}{'err roll':>10}{'err pitch':>11}")
    results = []
    for target in SEQUENCE_DEG:
        ctl.command_angles(math.radians(target), math.radians(target))
        time.sleep(SETTLE_S)
        a, b = settled()
        results.append((target, a, b))
        print(f"{target:8d}{a:10.3f}{b:10.3f}{a-target:10.3f}{b-target:11.3f}")
    return results


def spread_at_repeats(results):
    """Max minus min actual angle among visits to the same target."""
    by_target = {}
    for target, a, b in results:
        by_target.setdefault(target, []).append((a, b))
    out = {}
    for target, vals in by_target.items():
        if len(vals) < 2:
            continue
        out[target] = (max(v[0] for v in vals) - min(v[0] for v in vals),
                       max(v[1] for v in vals) - min(v[1] for v in vals))
    return out


def report(label, results):
    spreads = spread_at_repeats(results)
    errs_r = [abs(a - t) for t, a, _ in results]
    errs_p = [abs(b - t) for t, _, b in results]
    print(f"  {label}: mean |error| roll {statistics.fmean(errs_r):.3f} deg, "
          f"pitch {statistics.fmean(errs_p):.3f} deg")
    for target, (sr, sp) in sorted(spreads.items()):
        print(f"    spread at {target:+d} deg: roll {sr:.3f}  pitch {sp:.3f}")
    return statistics.fmean(errs_r), statistics.fmean(errs_p), spreads


def main():
    run = sys.argv[1]
    backlash = measure_backlash(run)
    print("lost motion from the unconditioned sweep (interior median):")
    for k, v in backlash.items():
        print(f"  {k}: {math.degrees(v):.3f} deg")

    contract = ServoContract.from_json(
        str(ROOT / "calib" / "servo_calibration.json"))
    limits = TiltLimits.from_json(
        str(ROOT / "calib" / "servo_travel_limits.json"))

    with BNO086Stream() as imu:
        imu.load_zero(str(ROOT / "calib" / "imu_zero.json"))
        threading.Thread(target=reader, args=(imu,), daemon=True).start()
        time.sleep(0.8)
        with STS3215Bus() as bus:
            for sid in (1, 2):
                bus.apply_config(sid)

            off = TiltController(bus, contract, limits)
            off.enable()
            res_off = run_sequence(off, "compensation OFF")

            on = TiltController(
                bus, contract, limits,
                roll_backlash=AxisBacklash(backlash["roll"], True),
                pitch_backlash=AxisBacklash(backlash["pitch"], True))
            on.enable()
            res_on = run_sequence(on, "compensation ON")

            print("\n== summary ==")
            e_off = report("OFF", res_off)
            e_on = report("ON ", res_on)
            print(f"\n  mean |error| roll  {e_off[0]:.3f} -> {e_on[0]:.3f} deg")
            print(f"  mean |error| pitch {e_off[1]:.3f} -> {e_on[1]:.3f} deg")
            bus.sync_write_positions({
                contract.roll.servo_id: contract.roll.center_counts,
                contract.pitch.servo_id: contract.pitch.center_counts,
            })
    stop.set()
    print("\n  Note: the first command of each run gets no correction -- "
          "direction is unknown until the axis has moved -- so exclude it when "
          "comparing spreads.")


if __name__ == "__main__":
    main()
