#!/usr/bin/env python3
"""System identification for the STS3215 tilt servos, measured against the IMU.

Produces the numbers ``contract/servo_contract.py`` declares but refuses to
guess: gain, offset, sign, deadband, backlash, latency, rise time, and rate
limit. Until this runs, ``AxisCalibration.measured`` is False and
``angle_to_counts`` raises by design -- you cannot calibrate the mapping using
the mapping, so everything here commands raw counts through the driver.

Four experiments, in a deliberate order:

    0  axis discovery      which servo drives which angle, sign, cross-coupling
    A  static sweep        counts -> angle gain and offset, plus linearity
    B  hysteresis          deadband and backlash from an up-then-down sweep
    C  step response       latency, rise time, and peak rate

0 runs first because nothing else is safe until the signs are known. A runs
before C because C jumps, and jumping needs a range that A has already proved.

Ground truth is the BNO086. The servo's own encoder is logged alongside it
throughout -- it is free on the same 1 Mbps bus, and where the two disagree the
difference is linkage compliance and backlash rather than gearbox error.

Examples:
    python3 tools/sysid_actuator.py --dry-run          # preflight only, no motion
    python3 tools/sysid_actuator.py --experiment 0     # axis discovery only
    python3 tools/sysid_actuator.py                    # all four
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tag_vision.hardware.imu import BNO086Stream, ImuError  # noqa: E402
from tag_vision.hardware.motion import ramp_to  # noqa: E402
from tag_vision.hardware.sts3215 import (  # noqa: E402
    Mode,
    Register,
    STS3215Bus,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZERO = ROOT / "calib" / "imu_zero.json"

SERVO_IDS = (1, 2)

# STS3215 is a 2S device: 6.0-8.4 V. servo_scan.json once recorded 8.6 V with a
# voltage flag set, and numbers measured over the limit do not describe the
# servo you will actually fly.
VOLTAGE_MIN_V = 6.0
VOLTAGE_MAX_V = 8.4


@dataclass
class AngleSample:
    host_time: float
    esp_micros: int
    alpha_rad: float
    beta_rad: float
    seq: int


@dataclass
class Settled:
    """One dwell point: commanded counts and the resulting settled angle."""

    servo_id: int
    commanded_counts: int
    encoder_counts: int
    alpha_rad: float
    beta_rad: float
    alpha_sd: float
    beta_sd: float
    samples: int
    peak_load: int
    direction: str = "up"


class ImuRecorder:
    """Background reader so IMU sampling is not gated by the servo bus.

    Necessary rather than tidy: a step response needs 200 Hz sampling *while*
    the servo bus is being polled, and doing both in one loop would put the bus
    round trip inside every IMU interval.
    """

    def __init__(self, imu: BNO086Stream) -> None:
        self.imu = imu
        self.samples: list[AngleSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.read_failures = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self.imu.read_sample(timeout=0.5)
            except Exception:  # noqa: BLE001 - counted, loop must survive
                self.read_failures += 1
                continue
            if sample is None:
                continue
            alpha, beta = self.imu.angles(sample)
            with self._lock:
                self.samples.append(AngleSample(
                    host_time=sample.host_time,
                    esp_micros=sample.esp_micros,
                    alpha_rad=alpha,
                    beta_rad=beta,
                    seq=sample.seq,
                ))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def mark(self) -> int:
        with self._lock:
            return len(self.samples)

    def since(self, index: int) -> list[AngleSample]:
        with self._lock:
            return list(self.samples[index:])

    def between(self, t0: float, t1: float) -> list[AngleSample]:
        with self._lock:
            return [s for s in self.samples if t0 <= s.host_time <= t1]

    def recent(self, count: int = 5) -> list[AngleSample]:
        with self._lock:
            return list(self.samples[-count:])

    def board_probe(self):
        """A ``ramp_to`` motion probe measuring how far the *board* has moved.

        Returns distance travelled in radians from wherever the board was when
        the probe was first called, so any movement in either axis registers.
        Passed to ``ramp_to`` so a stall means the plate stopped, not that the
        servo did -- servo 1 advanced 200 counts through its dead zone while
        the board moved 0.08 deg, and an encoder-based guard calls that healthy.
        """
        reference: tuple[float, float] | None = None

        def probe() -> float:
            nonlocal reference
            samples = self.recent(5)
            if not samples:
                return 0.0
            alpha = statistics.fmean(s.alpha_rad for s in samples)
            beta = statistics.fmean(s.beta_rad for s in samples)
            if reference is None:
                reference = (alpha, beta)
            return math.hypot(alpha - reference[0], beta - reference[1])

        return probe


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def linear_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Least squares y = slope*x + intercept. Returns (slope, intercept, rms).

    The residual is returned with the fit, not computed on request, because the
    slope alone is exactly what must not be trusted here: a servo horn driving a
    plate is a crank linkage and is only linear near centre.
    """
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    intercept = my - slope * mx
    rms = math.sqrt(
        sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y)) / n)
    return slope, intercept, rms


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def preflight(bus: STS3215Bus) -> dict:
    """Read and validate the conditions the measurements will be taken under.

    Recorded into the contract's ``servo_config``: these settings change what a
    step response looks like, so a fit measured under different gains is not
    comparable and must not be silently reused.
    """
    print("== preflight ==")
    configs = {}
    problems: list[str] = []

    for servo_id in SERVO_IDS:
        config = bus.read_config(servo_id)
        configs[str(servo_id)] = config

        voltage = config.get("present_voltage_v", float("nan"))
        mode = config.get("mode")
        print(f"  servo {servo_id}: {voltage:.1f} V, mode {mode}, "
              f"P {config.get('p_coefficient')} "
              f"I {config.get('i_coefficient')} "
              f"D {config.get('d_coefficient')}, "
              f"pos {bus.read_word(servo_id, Register.PRESENT_POSITION)}, "
              f"{config.get('present_temperature_c')} C")

        if not (VOLTAGE_MIN_V <= voltage <= VOLTAGE_MAX_V):
            problems.append(
                f"servo {servo_id} at {voltage:.1f} V, outside "
                f"{VOLTAGE_MIN_V}-{VOLTAGE_MAX_V} V")
        if mode != int(Mode.POSITION):
            problems.append(
                f"servo {servo_id} is in mode {mode}, not position (0). "
                "The contract's numbers describe position mode only.")
        temperature = config.get("present_temperature_c", 0)
        if temperature > 55:
            problems.append(f"servo {servo_id} at {temperature} C before starting")

    return {"configs": configs, "problems": problems}


# ---------------------------------------------------------------------------
# dwell
# ---------------------------------------------------------------------------
def dwell(
    bus: STS3215Bus,
    recorder: ImuRecorder,
    servo_id: int,
    target: int,
    *,
    settle_s: float,
    measure_s: float,
    direction: str = "up",
    max_load: int,
    ramp: int,
    condition_counts: int = 0,
    condition_cycles: int = 3,
) -> Settled | None:
    """Move to ``target``, wait out the transient, then average the angle.

    ``condition_counts`` makes every point be reached from the same side: the
    axis is driven that many counts below the target and brought back up, a few
    times. Servo 1 has real direction-dependent lost motion, and returning to
    one count from a fixed side settles to ±0.009 deg after about three cycles,
    against 0.73 deg when approach direction varies. Conditioning trades run
    time for that factor of seventy.

    Note what this costs: with every point conditioned identically, the up and
    down legs no longer differ, so experiment B measures residual hysteresis
    *under conditioning* rather than the raw lost motion. The unconditioned
    figures (0.734 deg alpha, 0.197 deg beta) are already recorded.
    """
    if condition_counts > 0:
        floor = max(0, target - condition_counts)
        for _ in range(max(1, condition_cycles)):
            approach = ramp_to(bus, servo_id, floor, max_load=max_load,
                               ramp=ramp, settle=0.05)
            if not approach.arrived:
                print(f"    conditioning aborted below {target}: "
                      f"load {approach.abort_load}")
                return None
            ramp_to(bus, servo_id, target, max_load=max_load, ramp=ramp,
                    settle=0.05)
    result = ramp_to(bus, servo_id, target, max_load=max_load,
                     ramp=ramp, settle=0.05,
                     motion_probe=recorder.board_probe(),
                     # 0.02 deg: comfortably above the sensor's 0.006 deg noise,
                     # far below the ~0.3 deg a healthy 50-count step produces.
                     motion_epsilon=math.radians(0.02))
    if not result.arrived:
        print(f"    ABORT at {target}: load {result.abort_load} "
              f"exceeded {max_load} near {result.aborted_at}")
        return None

    time.sleep(settle_s)
    start = time.time()
    time.sleep(measure_s)
    window = recorder.between(start, time.time())
    if not window:
        print(f"    no IMU samples during dwell at {target}")
        return None

    alpha, alpha_sd = mean_sd([s.alpha_rad for s in window])
    beta, beta_sd = mean_sd([s.beta_rad for s in window])
    return Settled(
        servo_id=servo_id,
        commanded_counts=target,
        encoder_counts=bus.read_word(servo_id, Register.PRESENT_POSITION),
        alpha_rad=alpha,
        beta_rad=beta,
        alpha_sd=alpha_sd,
        beta_sd=beta_sd,
        samples=len(window),
        peak_load=result.peak_load,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# experiment 0: axis discovery and cross-coupling
# ---------------------------------------------------------------------------
def experiment_axis_discovery(bus, recorder, centers, args) -> dict:
    """First motion of the rig. Which servo drives which angle, and how much.

    The cross-coupling matrix is the real output. ServoContract models two
    independent axes, one servo per angle; a stacked-hinge gimbal usually leaks,
    and if it does here then the contract's shape is wrong and no amount of
    careful fitting in the later experiments would reveal it.
    """
    print("\n== experiment 0: axis discovery ==")
    print(f"  +-{args.discovery_counts} counts per servo, load-guarded")

    jacobian: dict[str, dict[str, float]] = {}
    points: list[Settled] = []

    for servo_id in SERVO_IDS:
        center = centers[servo_id]
        low = center - args.discovery_counts
        high = center + args.discovery_counts
        print(f"  servo {servo_id}: {low} -> {high} (center {center})")

        readings = {}
        for label, target in (("low", low), ("center", center), ("high", high)):
            point = dwell(bus, recorder, servo_id, target,
                          settle_s=args.settle, measure_s=args.measure,
                          max_load=args.max_load, ramp=args.ramp,
                          condition_counts=args.condition_counts,
                          condition_cycles=args.condition_cycles)
            if point is None:
                print(f"  servo {servo_id} discovery aborted at {label}")
                return {"aborted": True, "servo_id": servo_id, "at": label,
                        "points": [asdict(p) for p in points]}
            readings[label] = point
            points.append(point)
            print(f"    {label:>6} {target:5d} -> alpha "
                  f"{math.degrees(point.alpha_rad):+7.3f} beta "
                  f"{math.degrees(point.beta_rad):+7.3f} deg")

        # Encoder travel, not commanded travel. Servo 1 sits at 4005 against a
        # 4095 ceiling, so a wide probe gets clamped; dividing by the commanded
        # span would then report a gain lower than the truth by whatever
        # fraction was never actually travelled.
        span = readings["high"].encoder_counts - readings["low"].encoder_counts
        if abs(span) < 1:
            raise RuntimeError(
                f"servo {servo_id} encoder moved {span} counts across the "
                "probe; it is not travelling and no gain can be derived")
        d_alpha = readings["high"].alpha_rad - readings["low"].alpha_rad
        d_beta = readings["high"].beta_rad - readings["low"].beta_rad
        jacobian[str(servo_id)] = {
            "d_alpha_per_count": d_alpha / span,
            "d_beta_per_count": d_beta / span,
        }

        # Return to centre before touching the other servo, so each column of
        # the Jacobian is measured with the other axis in the same place.
        dwell(bus, recorder, servo_id, center, settle_s=args.settle,
              measure_s=0.1, max_load=args.max_load, ramp=args.ramp,
              condition_counts=args.condition_counts,
              condition_cycles=args.condition_cycles)

    # Which angle does each servo predominantly drive?
    assignment = {}
    coupling_ratios = {}
    for servo_id in SERVO_IDS:
        j = jacobian[str(servo_id)]
        a, b = abs(j["d_alpha_per_count"]), abs(j["d_beta_per_count"])
        dominant = "alpha" if a >= b else "beta"
        assignment[str(servo_id)] = dominant
        weak, strong = (b, a) if dominant == "alpha" else (a, b)
        coupling_ratios[str(servo_id)] = (weak / strong) if strong > 0 else float("inf")

    print("\n  cross-coupling matrix (rad per count):")
    print(f"    {'':>10}{'d_alpha':>14}{'d_beta':>14}")
    for servo_id in SERVO_IDS:
        j = jacobian[str(servo_id)]
        print(f"    servo {servo_id}  {j['d_alpha_per_count']:14.3e}"
              f"{j['d_beta_per_count']:14.3e}")

    print("\n  assignment:")
    for servo_id in SERVO_IDS:
        ratio = coupling_ratios[str(servo_id)]
        print(f"    servo {servo_id} -> {assignment[str(servo_id)]} "
              f"(off-axis {ratio * 100:.1f}% of on-axis)")

    distinct = len(set(assignment.values())) == len(SERVO_IDS)
    worst_coupling = max(coupling_ratios.values())

    if not distinct:
        print("\n  WARNING: both servos map to the same angle. Either the "
              "linkage is not what the contract assumes, or one servo did not "
              "move. Do not proceed to experiments A-C on this result.")
    if worst_coupling > args.coupling_limit:
        print(f"\n  WARNING: cross-coupling {worst_coupling * 100:.1f}% exceeds "
              f"the {args.coupling_limit * 100:.0f}% limit. ServoContract models "
              "two independent axes; that assumption does not hold here and the "
              "contract needs a coupled form before any fit is meaningful.")

    return {
        "aborted": False,
        "jacobian_rad_per_count": jacobian,
        "assignment": assignment,
        "coupling_ratios": coupling_ratios,
        "axes_distinct": distinct,
        "worst_coupling": worst_coupling,
        "within_coupling_limit": worst_coupling <= args.coupling_limit,
        "points": [asdict(p) for p in points],
    }


# ---------------------------------------------------------------------------
# experiments A and B: static sweep and hysteresis
# ---------------------------------------------------------------------------
def experiment_sweep(bus, recorder, centers, assignment, args) -> dict:
    """Staircase up then back down. A is the up leg, B is the pair of legs."""
    print("\n== experiments A/B: static sweep and hysteresis ==")
    results: dict[str, dict] = {}

    for servo_id in SERVO_IDS:
        center = centers[servo_id]
        angle_key = assignment.get(str(servo_id), "alpha")
        half = args.sweep_counts
        step = args.sweep_step

        up = list(range(center - half, center + half + 1, step))
        down = list(reversed(up))
        print(f"\n  servo {servo_id} ({angle_key}): {up[0]} .. {up[-1]} "
              f"step {step}, {len(up)} points each way")

        points: list[Settled] = []
        aborted = False
        for direction, sequence in (("up", up), ("down", down)):
            for target in sequence:
                point = dwell(bus, recorder, servo_id, target,
                              settle_s=args.settle, measure_s=args.measure,
                              direction=direction, max_load=args.max_load,
                              ramp=args.ramp,
                              condition_counts=args.condition_counts,
                              condition_cycles=args.condition_cycles)
                if point is None:
                    aborted = True
                    break
                points.append(point)
                angle = point.alpha_rad if angle_key == "alpha" else point.beta_rad
                print(f"    {direction:>4} {target:5d} -> "
                      f"{math.degrees(angle):+7.3f} deg  "
                      f"enc {point.encoder_counts:5d}  load {point.peak_load}")
            if aborted:
                break

        # Park back at centre whatever happened.
        ramp_to(bus, servo_id, center, max_load=args.max_load, ramp=args.ramp)

        results[str(servo_id)] = _fit_sweep(points, angle_key, center, aborted)

    return results


def _fit_sweep(points: list[Settled], angle_key: str, center: int,
               aborted: bool) -> dict:
    def angle_of(p: Settled) -> float:
        return p.alpha_rad if angle_key == "alpha" else p.beta_rad

    up = [p for p in points if p.direction == "up"]
    down = [p for p in points if p.direction == "down"]

    summary: dict = {
        "angle": angle_key,
        "center_counts_commanded": center,
        "aborted": aborted,
        "points": [asdict(p) for p in points],
    }
    if len(up) < 3:
        summary["error"] = "too few points to fit"
        return summary

    # Fit angle as a function of counts, then invert: the contract stores
    # counts_per_rad, but counts is the independent variable here.
    counts = [float(p.commanded_counts) for p in up]
    angles = [angle_of(p) for p in up]
    slope, intercept, rms = linear_fit(counts, angles)

    if slope == 0 or math.isnan(slope):
        summary["error"] = "degenerate fit; did the servo move?"
        return summary

    counts_per_rad = abs(1.0 / slope)
    sign = 1 if slope > 0 else -1
    center_counts = -intercept / slope  # counts where the fitted angle is zero

    # Residual in degrees is the number that decides whether a straight line is
    # honest here.
    rms_deg = math.degrees(rms)
    angle_span_deg = math.degrees(max(angles) - min(angles))

    summary.update({
        "counts_per_rad": counts_per_rad,
        "sign": sign,
        "center_counts": center_counts,
        "fit_rms_rad": rms,
        "fit_rms_deg": rms_deg,
        "angle_span_deg": angle_span_deg,
        "linearity_pct": (rms_deg / angle_span_deg * 100.0)
        if angle_span_deg > 0 else float("nan"),
    })

    # Hysteresis: compare up and down at equal commanded counts.
    if down:
        by_counts_down = {p.commanded_counts: angle_of(p) for p in down}
        gaps = [abs(angle_of(p) - by_counts_down[p.commanded_counts])
                for p in up if p.commanded_counts in by_counts_down]
        if gaps:
            summary["backlash_rad"] = statistics.fmean(gaps)
            summary["backlash_deg"] = math.degrees(statistics.fmean(gaps))
            summary["backlash_max_deg"] = math.degrees(max(gaps))
            summary["deadband_counts"] = (
                statistics.fmean(gaps) * counts_per_rad)

    # Encoder-vs-IMU: where the servo says it moved but the board did not, the
    # slop is in the linkage rather than the gearbox.
    enc = [float(p.encoder_counts) for p in up]
    enc_slope, _, enc_rms = linear_fit(enc, angles)
    if enc_slope not in (0.0,) and not math.isnan(enc_slope):
        summary["counts_per_rad_encoder"] = abs(1.0 / enc_slope)
        summary["encoder_fit_rms_deg"] = math.degrees(enc_rms)

    return summary


# ---------------------------------------------------------------------------
# experiment C: step response
# ---------------------------------------------------------------------------
def experiment_steps(bus, recorder, centers, assignment, args, link_rtt_s) -> dict:
    """Unramped steps, measured at 200 Hz.

    This is the one place ramp_to is deliberately not used: a ramp is exactly
    what a step response must not contain. Amplitudes are limited to the range
    experiment A already traversed without tripping the load guard, and the load
    is checked immediately after each step.
    """
    print("\n== experiment C: step response ==")
    print(f"  de-biasing latency by half the {link_rtt_s * 1e3:.2f} ms link RTT")
    results: dict[str, dict] = {}

    for servo_id in SERVO_IDS:
        center = centers[servo_id]
        angle_key = assignment.get(str(servo_id), "alpha")
        steps = []

        for amplitude in args.step_counts:
            for sign in (1, -1):
                target = center + sign * amplitude
                print(f"  servo {servo_id}: step {sign * amplitude:+d} counts")

                ramp_to(bus, servo_id, center, max_load=args.max_load, ramp=args.ramp)
                time.sleep(args.settle)

                recorder.mark()
                bus.torque_enable(servo_id)
                command_time = time.time()
                bus.set_goal_position(servo_id, target)

                time.sleep(args.step_window)
                window = recorder.between(command_time - 0.05,
                                          command_time + args.step_window)

                state = bus.read_state(servo_id)
                if abs(state.load) > args.max_load:
                    print(f"    load {state.load} after step; backing off")
                    ramp_to(bus, servo_id, center, max_load=args.max_load, ramp=args.ramp)
                    steps.append({"amplitude_counts": sign * amplitude,
                                  "error": "load limit after step"})
                    continue

                metrics = _step_metrics(window, command_time, angle_key,
                                        link_rtt_s)
                metrics["amplitude_counts"] = sign * amplitude
                metrics["encoder_final"] = state.position
                steps.append(metrics)

                if "rise_time_s" in metrics:
                    print(f"    latency {metrics['step_latency_s'] * 1e3:6.1f} ms  "
                          f"rise {metrics['rise_time_s'] * 1e3:6.1f} ms  "
                          f"peak {math.degrees(metrics['max_rate_rad_s']):6.1f} deg/s")
                else:
                    print(f"    {metrics.get('error', 'no metrics')}")

        ramp_to(bus, servo_id, center, max_load=args.max_load, ramp=args.ramp)

        usable = [s for s in steps if "rise_time_s" in s]
        summary = {"steps": steps}
        if usable:
            summary.update({
                "step_latency_s": statistics.median(
                    [s["step_latency_s"] for s in usable]),
                "rise_time_s": statistics.median(
                    [s["rise_time_s"] for s in usable]),
                "max_rate_rad_s": max(s["max_rate_rad_s"] for s in usable),
            })
        results[str(servo_id)] = summary

    return results


def _step_metrics(window: list[AngleSample], command_time: float,
                  angle_key: str, link_rtt_s: float) -> dict:
    if len(window) < 5:
        return {"error": f"only {len(window)} samples in step window"}

    def angle_of(s: AngleSample) -> float:
        return s.alpha_rad if angle_key == "alpha" else s.beta_rad

    pre = [angle_of(s) for s in window if s.host_time < command_time]
    post = [s for s in window if s.host_time >= command_time]
    if len(pre) < 2 or len(post) < 5:
        return {"error": "insufficient pre/post samples"}

    start_angle = statistics.fmean(pre)
    # Final value from the tail, not the single last sample, so overshoot
    # ringing does not set the 90% threshold.
    tail_samples = post[-max(3, len(post) // 5):]
    tail = [angle_of(s) for s in tail_samples]
    final_angle = statistics.fmean(tail)
    delta = final_angle - start_angle

    if abs(delta) < math.radians(0.05):
        return {"error": "step produced no measurable movement",
                "delta_deg": math.degrees(delta)}

    # Is the board actually still at the end of the window?
    #
    # This check is not optional. Every threshold below is a fraction of
    # ``delta``, and ``delta`` is measured from the tail. If the window ended
    # while the board was still moving, delta is the truncated movement rather
    # than the real one, the 90% crossing lands early against it, and the
    # function returns a confidently wrong -- too fast -- rise time. Nothing
    # downstream could detect that, because the numbers look entirely ordinary.
    tail_slope, _, _ = linear_fit([s.host_time for s in tail_samples], tail)
    if not math.isnan(tail_slope):
        rise_span = post[-1].host_time - post[0].host_time
        mean_rate = abs(delta) / rise_span if rise_span > 0 else float("inf")
        if abs(tail_slope) > 0.10 * mean_rate:
            return {
                "error": "step had not settled when the window ended; "
                         "increase --step-window",
                "delta_deg": math.degrees(delta),
                "tail_rate_deg_s": math.degrees(tail_slope),
            }

    def crossing(fraction: float) -> float | None:
        threshold = start_angle + fraction * delta
        for sample in post:
            value = angle_of(sample)
            if (delta > 0 and value >= threshold) or (
                    delta < 0 and value <= threshold):
                return sample.host_time
        return None

    t5 = crossing(0.05)
    t10 = crossing(0.10)
    t90 = crossing(0.90)
    if t5 is None or t10 is None or t90 is None:
        return {"error": "step never reached 90% within the window",
                "delta_deg": math.degrees(delta)}

    # Half the round trip is the one-way transport delay this measurement
    # inherits from the USB link. Subtracting it is what makes the number a
    # property of the servo rather than of the cable.
    one_way = link_rtt_s / 2.0
    latency = max(0.0, (t5 - command_time) - one_way)

    rates = []
    for a, b in zip(post, post[1:]):
        dt = b.host_time - a.host_time
        if dt > 0:
            rates.append(abs(angle_of(b) - angle_of(a)) / dt)

    return {
        "step_latency_s": latency,
        "step_latency_raw_s": t5 - command_time,
        "link_one_way_s": one_way,
        "rise_time_s": t90 - t10,
        "max_rate_rad_s": max(rates) if rates else float("nan"),
        "delta_deg": math.degrees(delta),
        "samples": len(post),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--servo-port", default=None, help="auto-resolved by USB id; override only if needed")
    parser.add_argument("--imu-port", default=None, help="auto-resolved by USB id; override only if needed")
    parser.add_argument("--zero", type=Path, default=DEFAULT_ZERO)
    parser.add_argument("--experiment", choices=["0", "A", "B", "C", "all"],
                        default="all")
    parser.add_argument("--dry-run", action="store_true",
                        help="preflight and IMU check only; command no motion")

    parser.add_argument("--discovery-counts", type=int, default=50,
                        help="half-span for experiment 0 (default: %(default)s)")
    parser.add_argument("--sweep-counts", type=int, default=150,
                        help="half-span for the static sweep")
    parser.add_argument("--sweep-step", type=int, default=25)
    parser.add_argument("--step-counts", type=int, nargs="+",
                        default=[40, 80, 120])
    parser.add_argument("--step-window", type=float, default=1.0)

    parser.add_argument("--settle", type=float, default=0.4)
    parser.add_argument("--measure", type=float, default=0.4)

    # The plate is heavy, and that changes what these two must be.
    #
    # ``ramp_to`` steps toward the target in increments. With a 10-count
    # increment this rig does not break static friction: the servo strains
    # without moving and reports a stall load above its own 1000 limit. Measured
    # on servo 2 -- a +1 count move read load 1072 and aborted, while a +100
    # count move in 50-count increments read load 24 and tilted the board
    # 0.371 deg. The load figure was reporting stiction, not a jam.
    parser.add_argument("--ramp", type=int, default=50,
                        help="move increment in counts; too small stalls a "
                             "heavy plate (default: %(default)s)")
    parser.add_argument("--max-load", type=int, default=700,
                        help="abort above this load. Headroom to break stiction "
                             "while staying under the servo's own 1000 limit "
                             "(default: %(default)s)")
    parser.add_argument("--condition-counts", type=int, default=0,
                        help="approach every point from this many counts below, "
                             "so direction-dependent lost motion is not sampled "
                             "from both sides. 0 disables (default: %(default)s)")
    parser.add_argument("--condition-cycles", type=int, default=3,
                        help="conditioning approaches per point; the axis "
                             "settles after about three (default: %(default)s)")
    parser.add_argument("--coupling-limit", type=float, default=0.10,
                        help="off-axis/on-axis ratio above which the "
                             "independent-axis contract is considered invalid")
    parser.add_argument("--force", action="store_true",
                        help="run despite preflight problems (records them)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "artifacts" / "sysid" / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"output: {outdir}")

    # --- IMU -------------------------------------------------------------
    try:
        imu = BNO086Stream(port=args.imu_port)
    except Exception as exc:  # noqa: BLE001
        print(f"could not open IMU on {args.imu_port}: {exc}")
        return 1

    with imu:
        if not args.zero.exists():
            print(f"no level zero at {args.zero}. Run:\n"
                  f"  python3 tools/imu_monitor.py --capture-zero")
            return 1
        imu.load_zero(args.zero)
        zero_meta = json.loads(args.zero.read_text())
        print(f"loaded zero from {args.zero}")

        probe = imu.read_sample(timeout=2.0)
        if probe is None:
            print("no IMU samples; is the firmware running?")
            for message in imu.status_messages[-3:]:
                print(f"  firmware: {message}")
            return 1

        try:
            _, link_rtt = imu.estimate_clock_offset(samples=11)
        except ImuError as exc:
            print(f"could not measure link latency: {exc}")
            print("  step latency would silently include transport delay; "
                  "refusing to guess it")
            return 1
        print(f"link round-trip: {link_rtt * 1e3:.2f} ms")

        # --- servos ------------------------------------------------------
        with STS3215Bus(port=args.servo_port) as bus:
            checks = preflight(bus)
            if checks["problems"]:
                print("\npreflight problems:")
                for problem in checks["problems"]:
                    print(f"  - {problem}")
                if not args.force:
                    print("\nrefusing to measure under these conditions "
                          "(--force to override)")
                    return 1

            centers = {
                i: bus.read_word(i, Register.PRESENT_POSITION)
                for i in SERVO_IDS
            }
            # Prefer the counts recorded when the level zero was captured: the
            # zero and those counts are only meaningful as a pair.
            recorded = zero_meta.get("servo_counts_at_zero") or {}
            for servo_id in SERVO_IDS:
                if str(servo_id) in recorded:
                    was = recorded[str(servo_id)]
                    if was != centers[servo_id]:
                        print(f"  note: servo {servo_id} is at "
                              f"{centers[servo_id]}, but the zero was captured "
                              f"at {was}. Using {was} as centre.")
                    centers[servo_id] = was

            if args.dry_run:
                print("\ndry run: preflight passed, no motion commanded")
                return 0

            recorder = ImuRecorder(imu)
            recorder.start()
            results: dict = {
                "timestamp": stamp,
                "zero_file": str(args.zero),
                "centers": centers,
                "link_rtt_s": link_rtt,
                "preflight": checks,
                "args": vars(args) | {"step_counts": list(args.step_counts),
                                      "zero": str(args.zero)},
            }

            try:
                want = args.experiment
                discovery = experiment_axis_discovery(bus, recorder, centers, args)
                results["experiment_0"] = discovery

                if discovery.get("aborted"):
                    print("\nstopping: axis discovery aborted on a load limit")
                elif not discovery["axes_distinct"]:
                    print("\nstopping: the two servos do not drive distinct "
                          "angles, so per-axis fits would be meaningless")
                elif want in ("A", "B", "C", "all") and want != "0":
                    assignment = discovery["assignment"]
                    if want in ("A", "B", "all"):
                        results["experiment_AB"] = experiment_sweep(
                            bus, recorder, centers, assignment, args)
                    if want in ("C", "all"):
                        results["experiment_C"] = experiment_steps(
                            bus, recorder, centers, assignment, args, link_rtt)
            finally:
                for servo_id in SERVO_IDS:
                    try:
                        ramp_to(bus, servo_id, centers[servo_id],
                                max_load=args.max_load, ramp=args.ramp,
                          condition_counts=args.condition_counts,
                          condition_cycles=args.condition_cycles)
                        bus.torque_disable(servo_id)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  warning: could not park servo {servo_id}: {exc}")
                recorder.stop()

            results["imu_health"] = {
                "dropped": imu.dropped,
                "crc_errors": imu.crc_errors,
                "resyncs": imu.resyncs,
                "read_failures": recorder.read_failures,
                "total_samples": len(recorder.samples),
            }

            (outdir / "sysid.json").write_text(
                json.dumps(results, indent=2, default=str) + "\n",
                encoding="utf-8")
            with (outdir / "samples.jsonl").open("w", encoding="utf-8") as fh:
                for sample in recorder.samples:
                    fh.write(json.dumps(asdict(sample)) + "\n")

            print(f"\nwrote {outdir / 'sysid.json'}")
            print(f"wrote {outdir / 'samples.jsonl'} "
                  f"({len(recorder.samples)} samples)")
            print(f"IMU health: {imu.dropped} dropped, {imu.crc_errors} crc, "
                  f"{imu.resyncs} resyncs")
            print("\nNext: python3 tools/fit_sysid.py "
                  f"{outdir / 'sysid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
