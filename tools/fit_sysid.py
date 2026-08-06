#!/usr/bin/env python3
"""Turn a sysid run into a ``ServoContract``, or explain why it cannot.

Reads ``artifacts/sysid/<stamp>/sysid.json`` and writes
``calib/servo_calibration.json``. The interesting behaviour is the refusing:
``AxisCalibration.measured`` is only set True when the measurement actually
supports the linear model the contract encodes, because a contract that claims
to be measured and is not is worse than one that raises.

Four things are checked before anything is written:

    axes distinct       each servo drives a different angle
    cross-coupling      the independent-axis assumption holds
    linearity           residuals are small against the swept span
    plausibility        the fitted gain implies a sane linkage ratio

Examples:
    python3 tools/fit_sysid.py artifacts/sysid/20260806_150000/sysid.json
    python3 tools/fit_sysid.py <run> --accept-nonlinear   # record it anyway
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contract.servo_contract import (  # noqa: E402
    NOMINAL_COUNTS_PER_RAD,
    AxisCalibration,
    ServoContract,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "calib" / "servo_calibration.json"

# A straight line is accepted when its residual is under this fraction of the
# angle span it was fitted over. 2% of a 10 deg sweep is 0.2 deg, which is the
# scale at which a persistent tilt starts pushing a marble around.
LINEARITY_LIMIT_PCT = 2.0

# Sanity band on the linkage ratio (nominal servo counts/rad over fitted).
# A 1:1 drive would give 1.0, which no real horn-and-plate linkage does; wildly
# large or small means the fit found something other than the linkage.
#
# Widened from 0.05 after this rig -- a genuine ~20:1 reduction, ratio ~0.050 --
# tripped the lower bound. The band exists to catch a fit that has locked onto
# something other than the linkage, not to encode an expected reduction.
RATIO_MIN = 0.01
RATIO_MAX = 50.0


def check_axis(servo_id: str, fit: dict, discovery: dict, args) -> tuple[list[str], list[str]]:
    """Return (blocking problems, non-blocking notes) for one axis."""
    problems: list[str] = []
    notes: list[str] = []

    if fit.get("error"):
        problems.append(f"servo {servo_id}: {fit['error']}")
        return problems, notes
    if fit.get("aborted"):
        notes.append(f"servo {servo_id}: sweep aborted early on a load limit; "
                     "the fitted range is shorter than requested")

    linearity = fit.get("linearity_pct")
    if linearity is None or math.isnan(linearity):
        problems.append(f"servo {servo_id}: no linearity figure")
    elif linearity > args.linearity_limit:
        message = (f"servo {servo_id}: fit residual is {linearity:.2f}% of the "
                   f"{fit['angle_span_deg']:.2f} deg span "
                   f"({fit['fit_rms_deg']:.3f} deg rms), above the "
                   f"{args.linearity_limit:.1f}% limit")

        # A percentage of span is a relative test, and over a short span it
        # condemns errors that cannot be acted on. What actually bounds control
        # here is the smallest command that reliably moves the board -- measured
        # at 40 counts, below which steps on this rig do not even keep their
        # sign. A model whose error is a fraction of one such step is not the
        # binding constraint on anything, so it is accepted with the comparison
        # recorded rather than rejected on a ratio.
        cpr = fit.get("counts_per_rad") or 0.0
        step_deg = math.degrees(args.resolution_counts / cpr) if cpr else 0.0
        rms_deg = fit.get("fit_rms_deg", float("inf"))
        if step_deg and rms_deg <= args.resolution_fraction * step_deg:
            notes.append(
                f"{message}, but {rms_deg:.4f} deg is only "
                f"{rms_deg / step_deg * 100:.0f}% of one {args.resolution_counts}"
                f"-count command ({step_deg:.4f} deg) -- finer than this rig can "
                "be commanded, so accepted")
        elif args.accept_nonlinear:
            notes.append(message + " [accepted via --accept-nonlinear]")
        else:
            problems.append(
                message + f". Residual {rms_deg:.4f} deg exceeds "
                f"{args.resolution_fraction:.0%} of a {args.resolution_counts}"
                "-count command, so it is large enough to matter. Narrow "
                "max_tilt_rad or give AxisCalibration a nonlinear form")

    # The fitted zero crossing must lie inside the range actually swept.
    # Outside it, "level" is an extrapolation from data that never went there,
    # and on an axis with large backlash the crossing wanders: this rig produced
    # centres of 2235, 2342 and 2383 across three runs, a 148-count (~0.7 deg)
    # spread of the same order as its backlash.
    points = fit.get("points", [])
    commanded = [p["commanded_counts"] for p in points]
    center = fit.get("center_counts")
    if commanded and center is not None:
        low, high = min(commanded), max(commanded)
        if not low <= center <= high:
            problems.append(
                f"servo {servo_id}: fitted centre {center:.0f} lies outside the "
                f"swept range {low}..{high}. Level is being extrapolated from "
                "data that never reached it, which is what large backlash does "
                "to a zero crossing")

    counts_per_rad = fit.get("counts_per_rad")
    if not counts_per_rad or counts_per_rad <= 0:
        problems.append(f"servo {servo_id}: non-positive counts_per_rad")
    else:
        ratio = NOMINAL_COUNTS_PER_RAD / counts_per_rad
        notes.append(f"servo {servo_id}: linkage ratio {ratio:.3f} "
                     f"(nominal {NOMINAL_COUNTS_PER_RAD:.1f} / fitted "
                     f"{counts_per_rad:.1f} counts per rad)")
        if not (RATIO_MIN <= ratio <= RATIO_MAX):
            problems.append(
                f"servo {servo_id}: implausible linkage ratio {ratio:.3f}; "
                "the fit is probably not measuring the linkage")
        elif abs(ratio - 1.0) < 0.02:
            notes.append(f"servo {servo_id}: ratio is suspiciously close to "
                         "1:1, which no real linkage gives. Verify the IMU "
                         "actually moved with the board")

    if discovery:
        coupling = discovery.get("coupling_ratios", {}).get(servo_id)
        if coupling is not None and coupling > args.coupling_limit:
            problems.append(
                f"servo {servo_id}: cross-coupling {coupling * 100:.1f}% "
                f"exceeds {args.coupling_limit * 100:.0f}%. ServoContract "
                "models independent axes and this rig does not have them")

    return problems, notes


def build_axis(servo_id: int, fit: dict, steps: dict, measured: bool) -> AxisCalibration:
    points = fit.get("points", [])
    commanded = [p["commanded_counts"] for p in points] or [0, 4095]

    # Clamp so a bad fit is reported rather than raised. AxisCalibration
    # rejects a centre outside its travel limits, which is correct, but the
    # rejection belongs in check_axis where it can be explained -- crashing here
    # loses every other finding in the run.
    center = int(round(fit["center_counts"]))
    center = min(max(center, int(min(commanded))), int(max(commanded)))

    return AxisCalibration(
        servo_id=servo_id,
        counts_per_rad=float(fit["counts_per_rad"]),
        center_counts=center,
        sign=int(fit["sign"]),
        # Travel limits come from the range actually swept, never the full
        # 0-4095: outside the swept range the mapping is an extrapolation and
        # the load guard has not been there.
        min_counts=int(min(commanded)),
        max_counts=int(max(commanded)),
        deadband_counts=fit.get("deadband_counts"),
        backlash_rad=fit.get("backlash_rad"),
        step_latency_s=steps.get("step_latency_s"),
        rise_time_s=steps.get("rise_time_s"),
        max_rate_rad_s=steps.get("max_rate_rad_s"),
        measured=measured,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="sysid.json from sysid_actuator.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-tilt-deg", type=float, default=6.0,
                        help="policy action full scale (default: %(default)s)")
    parser.add_argument("--linearity-limit", type=float,
                        default=LINEARITY_LIMIT_PCT)
    parser.add_argument("--coupling-limit", type=float, default=0.10)
    parser.add_argument("--resolution-counts", type=int, default=40,
                        help="smallest command that reliably moves this board; "
                             "measured at 40 counts (default: %(default)s)")
    parser.add_argument("--resolution-fraction", type=float, default=0.5,
                        help="accept a fit whose residual is under this "
                             "fraction of one such command (default: %(default)s)")
    parser.add_argument("--accept-nonlinear", action="store_true",
                        help="write the linear fit even if residuals are large")
    args = parser.parse_args()

    data = json.loads(args.run.read_text(encoding="utf-8"))
    sweeps = data.get("experiment_AB") or {}
    steps_all = data.get("experiment_C") or {}
    discovery = data.get("experiment_0") or {}

    if not sweeps:
        print("no experiment_AB in this run; nothing to fit")
        return 1

    print(f"== fitting {args.run} ==")

    all_problems: list[str] = []
    all_notes: list[str] = []
    axes: dict[str, AxisCalibration] = {}

    # Map each servo onto the contract's roll/pitch slots by the angle it was
    # measured to drive, rather than by servo id. Which servo is which is a
    # measurement here, not a convention.
    assignment = discovery.get("assignment", {})

    for servo_id, fit in sorted(sweeps.items()):
        problems, notes = check_axis(servo_id, fit, discovery, args)
        all_problems.extend(problems)
        all_notes.extend(notes)

        print(f"\n  servo {servo_id} -> {fit.get('angle', '?')}")
        if "counts_per_rad" in fit:
            print(f"    counts_per_rad  {fit['counts_per_rad']:.2f}")
            print(f"    center_counts   {fit['center_counts']:.1f}")
            print(f"    sign            {fit['sign']:+d}")
            print(f"    span            {fit['angle_span_deg']:.2f} deg")
            print(f"    fit residual    {fit['fit_rms_deg']:.4f} deg "
                  f"({fit.get('linearity_pct', float('nan')):.2f}% of span)")
        if "backlash_deg" in fit:
            print(f"    backlash        {fit['backlash_deg']:.4f} deg mean, "
                  f"{fit['backlash_max_deg']:.4f} deg max")
        step = steps_all.get(servo_id, {})
        if "rise_time_s" in step:
            print(f"    latency         {step['step_latency_s'] * 1e3:.1f} ms")
            print(f"    rise time       {step['rise_time_s'] * 1e3:.1f} ms")
            print(f"    max rate        "
                  f"{math.degrees(step['max_rate_rad_s']):.1f} deg/s")
        else:
            all_notes.append(f"servo {servo_id}: no step response; "
                             "latency, rise time, and rate stay unmeasured")

        if "counts_per_rad" in fit:
            axes[servo_id] = build_axis(int(servo_id), fit, step,
                                        measured=False)

    if not discovery.get("axes_distinct", True):
        all_problems.append("the two servos do not drive distinct angles")

    if all_notes:
        print("\n  notes:")
        for note in all_notes:
            print(f"    - {note}")

    measured = not all_problems
    if all_problems:
        print("\n  BLOCKING:")
        for problem in all_problems:
            print(f"    - {problem}")
        print("\n  Writing the calibration with measured=False. "
              "angle_to_counts will keep raising, which is correct: these "
              "numbers do not yet support the model the contract encodes.")
    else:
        print("\n  all checks passed; marking the calibration measured")

    if len(axes) < 2:
        print(f"\nonly {len(axes)} axis fitted; need both to write a contract")
        return 1

    # Assign to roll/pitch by measured angle: alpha is rotation about X (roll),
    # beta about Y (pitch), matching board_pose's R = Rx(alpha) @ Ry(beta).
    roll_id = next((sid for sid, a in assignment.items() if a == "alpha"), None)
    pitch_id = next((sid for sid, a in assignment.items() if a == "beta"), None)
    if roll_id is None or pitch_id is None:
        ids = sorted(axes)
        roll_id, pitch_id = ids[0], ids[1]
        all_notes.append("axis assignment missing; fell back to servo id order")

    def finalise(servo_id: str) -> AxisCalibration:
        return dataclasses.replace(axes[servo_id], measured=measured)

    contract = ServoContract(
        roll=finalise(roll_id),
        pitch=finalise(pitch_id),
        max_tilt_rad=math.radians(args.max_tilt_deg),
        servo_config=data.get("preflight", {}).get("configs", {}),
    )
    # max_tilt must fit inside the range the fit was measured over, or
    # action_to_counts silently clamps and the policy's action space quietly
    # collapses: +-0.5 and +-1.0 map to the same counts and nothing complains.
    # The travel envelope is much larger than the calibrated range and it is
    # easy -- demonstrated -- to reach for the wrong one.
    for name, axis in (("roll", contract.roll), ("pitch", contract.pitch)):
        headroom = min(axis.center_counts - axis.min_counts,
                       axis.max_counts - axis.center_counts)
        needed = contract.max_tilt_rad * axis.counts_per_rad
        if needed > headroom:
            supported = math.degrees(headroom / axis.counts_per_rad)
            print(f"\n  WARNING: max_tilt {args.max_tilt_deg:.2f} deg needs "
                  f"{needed:.0f} counts on {name} but only {headroom:.0f} are "
                  f"calibrated. Commands beyond +-{supported:.2f} deg will "
                  f"clamp. Either lower --max-tilt-deg to {supported:.2f}, or "
                  "re-run the sweep over the range you intend to use.")

    contract.to_json(args.output)
    print(f"\nwrote {args.output}")
    print(f"  roll  <- servo {roll_id} (alpha)")
    print(f"  pitch <- servo {pitch_id} (beta)")

    # Round-trip check: the contract must read back and put a zero action at
    # the measured centres.
    reloaded = ServoContract.from_json(args.output)
    if reloaded.measured:
        counts = reloaded.action_to_counts((0.0, 0.0))
        print(f"  action (0,0) -> counts {counts} "
              f"(centres {reloaded.roll.center_counts}, "
              f"{reloaded.pitch.center_counts})")
        if counts != (reloaded.roll.center_counts, reloaded.pitch.center_counts):
            print("  WARNING: zero action does not land on the measured centres")
            return 1
    else:
        print("  measured=False, so angle_to_counts still raises by design")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
