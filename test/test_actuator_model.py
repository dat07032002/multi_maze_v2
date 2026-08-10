"""M1 acceptance: does the modelled actuator behave like the measured one?

The highest-value tests in the project, because they are the only checks against
real hardware numbers that cost no rig time. Everything here is compared to a
figure recorded in ``calib/servo_calibration.json`` or
``docs/PROJECT_STATUS.md``, never to a number invented in this file.
"""
from __future__ import annotations

import math
import statistics

import pytest

from sim.actuator import ActuatorModel
from sim.mjcf_builder import load_parameters

DT = 0.001

# tools/validate_backlash.py, run on the rig.
VALIDATE_SEQUENCE_DEG = [0, 2, 0, -2, 0, 2, 0, -2, 0]
SETTLE_S = 1.4
MEASURED = {
    "off": {"roll": 0.876, "pitch": 0.427},
    "on": {"roll": 0.261, "pitch": 0.258},
}


def step_response(model, index, amplitude_rad, seconds=1.5):
    """Latency to the 5 % crossing and the 10-90 % rise, as sysid defines them."""
    for _ in range(400):                      # settle, and take up the slack
        model.step(0.0, 0.0)
    start = model.angles[index]

    trace = [model.step(amplitude_rad, amplitude_rad)[index]
             for _ in range(int(seconds / DT))]
    delta = trace[-1] - start

    def crossing(fraction):
        threshold = start + fraction * delta
        for i, value in enumerate(trace):
            if value >= threshold:
                return i * DT
        raise AssertionError(f"never reached {fraction:.0%}")

    return crossing(0.05), crossing(0.90) - crossing(0.10)


def run_validate_sequence(compensate):
    model = ActuatorModel(DT, compensate=compensate)
    model.reset(0.0, 0.0)
    results = []
    for target_deg in VALIDATE_SEQUENCE_DEG:
        target = math.radians(target_deg)
        for _ in range(int(SETTLE_S / DT)):
            roll, pitch = model.step(target, target)
        results.append((target_deg, math.degrees(roll), math.degrees(pitch)))
    return {
        "roll": statistics.fmean(abs(a - t) for t, a, _ in results),
        "pitch": statistics.fmean(abs(b - t) for t, _, b in results),
        "results": results,
    }


# -- step response -----------------------------------------------------------
@pytest.mark.parametrize("axis,index", [("roll", 0), ("pitch", 1)])
def test_step_response_reproduces_the_recorded_latency_and_rise(axis, index):
    """Against a 80-count step, one of the amplitudes sysid actually used.

    Amplitude matters: 40/80/120 counts are all under the ~0.81 deg where the
    rate limit starts to bind, so this exercises the first-order lag rather
    than the saturation, which is what the recorded rise time describes.
    """
    params = load_parameters()
    model = ActuatorModel(DT, params=params)
    latency, rise = step_response(model, index, math.radians(0.389))

    expected_latency = params[f"actuator.{axis}.step_latency"]
    expected_rise = params[f"actuator.{axis}.rise_time"]
    assert latency == pytest.approx(expected_latency, rel=0.10), (
        f"{axis} 5% crossing {latency * 1000:.1f} ms vs recorded "
        f"{expected_latency * 1000:.1f} ms")
    assert rise == pytest.approx(expected_rise, rel=0.10), (
        f"{axis} 10-90% rise {rise * 1000:.1f} ms vs recorded "
        f"{expected_rise * 1000:.1f} ms")


def test_recorded_latency_is_the_five_percent_crossing_not_the_dead_time():
    """Feeding step_latency_s in as pure dead time would be wrong by ~6 ms.

    sysid defines latency as the 5 % crossing, so a first-order response spends
    tau*ln(1/0.95) of it merely climbing to 5 %. The model stores the pure dead
    time and reproduces the recorded figure; this pins that they differ.
    """
    params = load_parameters()
    model = ActuatorModel(DT, params=params)
    for axis in ("roll", "pitch"):
        dead = model.dynamics[axis].dead_time_s
        recorded = params[f"actuator.{axis}.step_latency"]
        assert dead < recorded
        assert recorded - dead == pytest.approx(
            model.dynamics[axis].tau_s * math.log(1 / 0.95), rel=1e-6)


def test_plate_does_not_move_at_all_during_the_dead_time():
    model = ActuatorModel(DT)
    model.reset(0.0, 0.0)
    dead = model.dynamics["roll"].dead_time_s
    for _ in range(int(dead / DT) - 1):
        roll, _ = model.step(math.radians(2.0), math.radians(2.0))
    assert abs(roll - model.dynamics["roll"].centre_bias_rad) < 1e-12


def test_rate_limit_saturates_at_the_measured_peak_rate():
    """Binds only for large steps -- above ~0.81 deg -- which is where a policy
    commanding the full +-4 deg range lives."""
    params = load_parameters()
    model = ActuatorModel(DT)
    model.reset(0.0, 0.0)
    previous = model.angles[0]
    peak = 0.0
    for _ in range(int(2.0 / DT)):
        roll, _ = model.step(params["actuator.max_tilt"], 0.0)
        peak = max(peak, abs(roll - previous) / DT)
        previous = roll
    assert peak == pytest.approx(params["actuator.roll.max_rate"], rel=0.02)


def test_slew_limit_is_invisible_at_the_amplitudes_sysid_measured():
    """Which is exactly why it cannot be treated as measured.

    sysid only commanded unramped steps of 40/80/120 counts. Across the whole
    plausible range of slew ceilings the step response at those amplitudes is
    identical, so the measurement contains no information about it.
    """
    params = load_parameters()
    responses = []
    for scale in (1.0, 5.0):
        params["actuator.slew_limit_scale"] = scale
        model = ActuatorModel(DT, params=params)
        responses.append(step_response(model, 0, math.radians(0.584)))
    assert responses[0] == pytest.approx(responses[1], abs=1e-9), (
        "the slew ceiling changed the response at a sysid amplitude, so this "
        "test no longer demonstrates what it claims")


def test_slew_limit_is_worth_a_dead_time_at_policy_amplitudes():
    """And why it has to be randomised until it is measured.

    A 4 deg step -- ordinary for a policy on a +-4 deg action space -- takes
    275 ms longer at the observed rate than unclamped. That is comparable to
    the entire 185 ms dead time, and no measurement constrains it.
    """
    params = load_parameters()
    times = []
    for scale in (1.0, 5.0):
        params["actuator.slew_limit_scale"] = scale
        model = ActuatorModel(DT, params=params, compensate=False)
        model.reset(0.0, 0.0)
        for _ in range(400):
            model.step(0.0, 0.0)
        start = model.angles[0]
        trace = [model.step(math.radians(4.0), math.radians(4.0))[0]
                 for _ in range(3000)]
        threshold = start + 0.9 * (trace[-1] - start)
        times.append(next(i * DT for i, v in enumerate(trace) if v >= threshold))
    assert times[0] - times[1] > 0.2, (
        f"clamped {times[0] * 1000:.0f} ms vs unclamped {times[1] * 1000:.0f} ms; "
        "expected the ceiling to matter by more than 200 ms here")


def test_a_change_below_the_deadband_does_not_move_the_plate():
    """Pitch's deadband is 27.9 counts, four times roll's 6.4."""
    model = ActuatorModel(DT, compensate=False)
    model.reset(0.0, 0.0)
    for _ in range(2000):
        model.step(0.0, 0.0)
    settled = model.angles

    per_count = 1.0 / model.contract.pitch.counts_per_rad
    nudge = 20 * per_count            # under pitch's deadband, over roll's
    for _ in range(2000):
        roll, pitch = model.step(nudge, nudge)
    assert pitch == pytest.approx(settled[1], abs=1e-9)
    assert roll != pytest.approx(settled[0], abs=1e-9)


# -- backlash, against tools/validate_backlash.py ----------------------------
def test_uncompensated_roll_error_matches_the_rig():
    """The headline check. 0.876 deg measured on the bench.

    Pure backlash alone predicts 5.4/9 = 0.60 deg; the rest is the residual
    centre bias, which is why the model carries one.
    """
    error = run_validate_sequence(compensate=False)["roll"]
    assert error == pytest.approx(MEASURED["off"]["roll"], rel=0.10), (
        f"modelled {error:.3f} deg vs measured {MEASURED['off']['roll']:.3f}")


def test_compensated_roll_error_matches_the_rig():
    error = run_validate_sequence(compensate=True)["roll"]
    assert error == pytest.approx(MEASURED["on"]["roll"], rel=0.15), (
        f"modelled {error:.3f} deg vs measured {MEASURED['on']['roll']:.3f}")


def test_compensation_removes_most_of_the_roll_error():
    """Measured on the rig as 0.876 -> 0.261, a 3.4x improvement."""
    off = run_validate_sequence(compensate=False)["roll"]
    on = run_validate_sequence(compensate=True)["roll"]
    assert off / on > 3.0, f"compensation only improved {off:.3f} -> {on:.3f}"


def test_pitch_backlash_errs_toward_too_much_never_too_little():
    """A known limitation, and deliberately left in the safe direction.

    Pitch's real backlash is position-dependent -- about 1.5 deg mid-range
    against 0.4-0.6 near the extremes -- while the model uses the flat 1.35 deg
    that PROJECT_STATUS mandates for simulation. The validate sequence commands
    +-2 deg, out toward the extremes, so the model overestimates: 0.62 deg
    against 0.427 measured, an effective 0.91 deg of real backlash there.

    Overestimating is the right way to be wrong. A policy trained against more
    lost motion than the rig has is over-prepared; one trained against less
    would arrive expecting authority that is not there. Tabulating pitch's
    position dependence is on the M5 list.
    """
    error = run_validate_sequence(compensate=False)["pitch"]
    assert error > MEASURED["off"]["pitch"], (
        "model no longer overestimates pitch backlash -- if pitch's position "
        "dependence has been tabulated, update this test")
    assert error == pytest.approx(0.622, rel=0.05), (
        f"pitch error moved to {error:.3f} deg; expected the documented 0.622")


def test_a_reversal_costs_a_full_band_of_servo_travel():
    """~277 counts on roll before the plate begins to move at all.

    This is the behaviour the action-rate penalty exists to discourage, so it
    has to be present in the model rather than compensated away in it.
    """
    model = ActuatorModel(DT, compensate=False)
    model.reset(0.0, 0.0)
    for _ in range(3000):
        model.step(math.radians(2.0), math.radians(2.0))
    top = model.angles[0]

    counts_moved = 0
    plant = model.plants["roll"]
    shaft_at_reversal = plant.shaft_rad
    for _ in range(4000):
        roll, _ = model.step(0.0, 0.0)
        if roll < top - 1e-9:
            counts_moved = abs(plant.shaft_rad - shaft_at_reversal) * \
                model.contract.roll.counts_per_rad
            break
    assert counts_moved == pytest.approx(277, rel=0.15), (
        f"plate started moving after {counts_moved:.0f} counts of shaft "
        "travel, expected about 277")


def test_a_reversal_smaller_than_the_band_moves_the_plate_not_at_all():
    """The sharpest statement of why dithering is expensive on this rig.

    Go to +2 deg, then ask for +1. That is a 1 deg reversal against 1.35 deg of
    lost motion, so the servo turns roughly 200 counts and the plate does not
    move by a single microradian. A policy that oscillates around a setpoint
    would do this continuously -- lots of servo wear, no authority -- which is
    what the action-rate penalty in the reward is priced against.
    """
    model = ActuatorModel(DT, compensate=False)
    model.reset(0.0, 0.0)
    for _ in range(3000):
        model.step(math.radians(2.0), math.radians(2.0))
    top = model.angles[0]
    shaft_before = model.plants["roll"].shaft_rad

    for _ in range(3000):
        roll, _ = model.step(math.radians(1.0), math.radians(1.0))

    shaft_travel = abs(model.plants["roll"].shaft_rad - shaft_before) * \
        model.contract.roll.counts_per_rad
    # 1e-9 rad is 6e-8 degrees: zero by any physical measure, and above the
    # float noise the min/max clamp leaves behind.
    assert roll == pytest.approx(top, abs=1e-9), (
        f"plate moved {math.degrees(abs(roll - top)):.4f} deg on a reversal "
        "smaller than the backlash band")
    assert shaft_travel > 150, (
        f"servo only turned {shaft_travel:.0f} counts; expected it to move "
        "well into the slack while achieving nothing")
