"""Synthetic tests for the sysid fits.

Each test generates data from known parameters and asserts the fit recovers
them. That direction matters: a fit that runs without crashing on real data
proves nothing, because there is no independent truth to compare it against.
Here the truth is constructed, so a sign error or an inverted gain has nowhere
to hide.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fit_sysid  # noqa: E402
from sysid_actuator import (  # noqa: E402
    AngleSample,
    Settled,
    _fit_sweep,
    _step_metrics,
    linear_fit,
)

# Truth used across the sweep tests.
TRUE_COUNTS_PER_RAD = 900.0
TRUE_CENTER = 2048
TRUE_SIGN = -1


def make_point(counts: int, angle: float, direction: str = "up",
               encoder: int | None = None) -> Settled:
    return Settled(
        servo_id=1,
        commanded_counts=counts,
        encoder_counts=counts if encoder is None else encoder,
        alpha_rad=angle,
        beta_rad=0.0,
        alpha_sd=1e-5,
        beta_sd=1e-5,
        samples=80,
        peak_load=40,
        direction=direction,
    )


def sweep_points(backlash_rad: float = 0.0) -> list[Settled]:
    """A linear axis, optionally with a constant hysteresis gap."""
    counts = list(range(TRUE_CENTER - 150, TRUE_CENTER + 151, 25))
    points = []
    for c in counts:
        angle = TRUE_SIGN * (c - TRUE_CENTER) / TRUE_COUNTS_PER_RAD
        points.append(make_point(c, angle, "up"))
    for c in reversed(counts):
        angle = TRUE_SIGN * (c - TRUE_CENTER) / TRUE_COUNTS_PER_RAD
        points.append(make_point(c, angle + backlash_rad, "down"))
    return points


class TestLinearFit(unittest.TestCase):
    def test_recovers_slope_and_intercept(self):
        slope, intercept, rms = linear_fit([0, 1, 2, 3], [3, 5, 7, 9])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 3.0)
        self.assertAlmostEqual(rms, 0.0)

    def test_degenerate_inputs_are_nan_not_exceptions(self):
        """A single point or a constant x must not raise mid-experiment."""
        for x, y in ([1], [1]), ([2, 2, 2], [1, 2, 3]):
            slope, _, _ = linear_fit(x, y)
            self.assertTrue(math.isnan(slope))

    def test_residual_reports_curvature(self):
        # y = x^2 over [0,4] is badly served by a line; the residual must say so.
        xs = [0, 1, 2, 3, 4]
        _, _, rms = linear_fit(xs, [x * x for x in xs])
        self.assertGreater(rms, 1.0)


class TestSweepFit(unittest.TestCase):
    def test_recovers_gain_sign_and_center(self):
        fit = _fit_sweep(sweep_points(), "alpha", TRUE_CENTER, aborted=False)
        self.assertAlmostEqual(fit["counts_per_rad"], TRUE_COUNTS_PER_RAD,
                               places=6)
        self.assertEqual(fit["sign"], TRUE_SIGN)
        self.assertAlmostEqual(fit["center_counts"], TRUE_CENTER, places=6)

    def test_negative_sign_is_not_absorbed_into_the_gain(self):
        """counts_per_rad must stay positive; direction lives in sign.

        AxisCalibration.__post_init__ rejects a non-positive gain, so folding
        the direction into it would make the contract unconstructable.
        """
        fit = _fit_sweep(sweep_points(), "alpha", TRUE_CENTER, aborted=False)
        self.assertGreater(fit["counts_per_rad"], 0)
        self.assertIn(fit["sign"], (1, -1))

    def test_linear_data_reports_near_zero_residual(self):
        fit = _fit_sweep(sweep_points(), "alpha", TRUE_CENTER, aborted=False)
        self.assertLess(fit["linearity_pct"], 0.01)

    def test_curved_linkage_shows_up_in_the_residual(self):
        """A crank linkage is only linear near centre; the fit must expose it.

        This is the case that decides whether AxisCalibration's straight line is
        honest for this rig, so it must not be silently absorbed.
        """
        points = []
        for c in range(TRUE_CENTER - 150, TRUE_CENTER + 151, 25):
            offset = (c - TRUE_CENTER) / TRUE_COUNTS_PER_RAD
            # Add a quadratic term: sin-like horn travel, exaggerated.
            angle = TRUE_SIGN * (offset + 4.0 * offset ** 2)
            points.append(make_point(c, angle, "up"))
        fit = _fit_sweep(points, "alpha", TRUE_CENTER, aborted=False)
        self.assertGreater(fit["linearity_pct"], 2.0)

    def test_recovers_backlash_from_hysteresis(self):
        backlash = math.radians(0.4)
        fit = _fit_sweep(sweep_points(backlash), "alpha", TRUE_CENTER,
                         aborted=False)
        self.assertAlmostEqual(fit["backlash_rad"], backlash, places=6)
        self.assertAlmostEqual(fit["deadband_counts"],
                               backlash * TRUE_COUNTS_PER_RAD, places=4)

    def test_no_hysteresis_gives_zero_backlash(self):
        fit = _fit_sweep(sweep_points(0.0), "alpha", TRUE_CENTER, aborted=False)
        self.assertAlmostEqual(fit["backlash_rad"], 0.0, places=9)

    def test_flat_response_is_an_error_not_a_fit(self):
        """A servo that did not move must not produce a confident calibration."""
        points = [make_point(c, 0.0, "up")
                  for c in range(TRUE_CENTER - 100, TRUE_CENTER + 101, 25)]
        fit = _fit_sweep(points, "alpha", TRUE_CENTER, aborted=False)
        self.assertIn("error", fit)

    def test_too_few_points_is_an_error(self):
        fit = _fit_sweep([make_point(2048, 0.0)], "alpha", 2048, aborted=False)
        self.assertIn("error", fit)


class TestStepMetrics(unittest.TestCase):
    @staticmethod
    def window(latency_s: float, tau_s: float, amplitude_rad: float,
               rate_hz: float = 200.0, duration_s: float = 1.0,
               command_time: float = 1000.0) -> list[AngleSample]:
        """First-order step: dead time, then an exponential approach."""
        samples = []
        n = int(duration_s * rate_hz)
        for i in range(-10, n):
            t = command_time + i / rate_hz
            dt = t - command_time
            if dt < latency_s:
                angle = 0.0
            else:
                angle = amplitude_rad * (1.0 - math.exp(-(dt - latency_s) / tau_s))
            samples.append(AngleSample(host_time=t, esp_micros=int(t * 1e6),
                                       alpha_rad=angle, beta_rad=0.0, seq=i % 256))
        return samples

    def test_recovers_latency_and_rise_time(self):
        latency, tau = 0.030, 0.050
        window = self.window(latency, tau, math.radians(5.0))
        metrics = _step_metrics(window, 1000.0, "alpha", link_rtt_s=0.0)

        # 5% crossing lands within one 200 Hz sample of the true dead time.
        self.assertAlmostEqual(metrics["step_latency_s"], latency, delta=0.008)
        # First-order 10-90% rise is tau*ln(9).
        self.assertAlmostEqual(metrics["rise_time_s"], tau * math.log(9),
                               delta=0.012)
        self.assertGreater(metrics["max_rate_rad_s"], 0)

    def test_subtracts_one_way_link_delay(self):
        """Latency must be a servo property, not a property of the USB cable."""
        window = self.window(0.030, 0.050, math.radians(5.0))
        rtt = 0.010
        metrics = _step_metrics(window, 1000.0, "alpha", link_rtt_s=rtt)
        self.assertAlmostEqual(
            metrics["step_latency_raw_s"] - metrics["step_latency_s"],
            rtt / 2.0, places=6)

    def test_latency_never_goes_negative(self):
        """An RTT larger than the measured delay must clamp, not wrap."""
        window = self.window(0.001, 0.050, math.radians(5.0))
        metrics = _step_metrics(window, 1000.0, "alpha", link_rtt_s=0.5)
        self.assertGreaterEqual(metrics["step_latency_s"], 0.0)

    def test_no_movement_is_reported_as_an_error(self):
        window = self.window(0.03, 0.05, 0.0)
        metrics = _step_metrics(window, 1000.0, "alpha", link_rtt_s=0.0)
        self.assertIn("error", metrics)

    def test_truncated_step_is_an_error_not_a_wrong_rise_time(self):
        """If the window ends before 90%, say so rather than report a number."""
        window = self.window(0.03, 2.0, math.radians(5.0), duration_s=0.2)
        metrics = _step_metrics(window, 1000.0, "alpha", link_rtt_s=0.0)
        self.assertIn("error", metrics)

    def test_too_few_samples_is_an_error(self):
        metrics = _step_metrics([], 1000.0, "alpha", link_rtt_s=0.0)
        self.assertIn("error", metrics)


class TestAcceptance(unittest.TestCase):
    """The gate that decides whether measured=True is justified."""

    @staticmethod
    def args(**overrides):
        class Args:
            linearity_limit = 2.0
            coupling_limit = 0.10
            accept_nonlinear = False
            # Smallest command that reliably moves this board, and the share of
            # it a model error may occupy before it is worth blocking on.
            resolution_counts = 40
            resolution_fraction = 0.5
        args = Args()
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    @staticmethod
    def good_fit(**overrides) -> dict:
        fit = {
            "counts_per_rad": 900.0,
            "center_counts": 2048.0,
            "sign": -1,
            "angle_span_deg": 19.0,
            "fit_rms_deg": 0.02,
            "linearity_pct": 0.1,
            "points": [{"commanded_counts": c}
                       for c in range(1898, 2199, 25)],
        }
        fit.update(overrides)
        return fit

    def test_clean_fit_passes(self):
        problems, _ = fit_sysid.check_axis(
            "1", self.good_fit(),
            {"coupling_ratios": {"1": 0.01}}, self.args())
        self.assertEqual(problems, [])

    def test_nonlinear_fit_blocks(self):
        problems, _ = fit_sysid.check_axis(
            "1", self.good_fit(linearity_pct=8.0, fit_rms_deg=1.5),
            {"coupling_ratios": {"1": 0.01}}, self.args())
        self.assertTrue(any("linear" in p or "residual" in p for p in problems))

    def test_nonlinear_can_be_accepted_explicitly(self):
        problems, notes = fit_sysid.check_axis(
            "1", self.good_fit(linearity_pct=8.0, fit_rms_deg=1.5),
            {"coupling_ratios": {"1": 0.01}},
            self.args(accept_nonlinear=True))
        self.assertEqual(problems, [])
        self.assertTrue(any("accept-nonlinear" in n for n in notes))

    def test_residual_below_command_resolution_is_accepted(self):
        """A model error finer than one commandable step must not block.

        The percentage-of-span test is relative, and over a short span it
        condemns errors nothing can act on. This rig measured 0.071 deg of
        residual against a 40-count command worth 0.175 deg -- 41% of the
        smallest move it can make -- which is not the binding constraint on
        anything.
        """
        fit = self.good_fit(linearity_pct=2.4, fit_rms_deg=0.071,
                            counts_per_rad=13109.9)
        problems, notes = fit_sysid.check_axis(
            "1", fit, {"coupling_ratios": {"1": 0.01}}, self.args())
        self.assertEqual(problems, [])
        self.assertTrue(any("finer than this rig can be commanded" in n
                            for n in notes))

    def test_residual_above_command_resolution_still_blocks(self):
        """The escape hatch must not swallow an error that does matter."""
        # 40 counts at 900 counts/rad is 2.55 deg; half of that is 1.27 deg.
        fit = self.good_fit(linearity_pct=8.0, fit_rms_deg=1.5,
                            counts_per_rad=900.0)
        problems, _ = fit_sysid.check_axis(
            "1", fit, {"coupling_ratios": {"1": 0.01}}, self.args())
        self.assertTrue(any("large enough to matter" in p for p in problems))

    def test_cross_coupling_blocks(self):
        """Coupled axes invalidate the contract's shape, not just its numbers."""
        problems, _ = fit_sysid.check_axis(
            "1", self.good_fit(),
            {"coupling_ratios": {"1": 0.45}}, self.args())
        self.assertTrue(any("coupling" in p for p in problems))

    def test_implausible_linkage_ratio_blocks(self):
        problems, _ = fit_sysid.check_axis(
            "1", self.good_fit(counts_per_rad=0.5),
            {"coupling_ratios": {"1": 0.01}}, self.args())
        self.assertTrue(any("linkage ratio" in p for p in problems))

    def test_sweep_error_propagates(self):
        problems, _ = fit_sysid.check_axis(
            "1", {"error": "degenerate fit; did the servo move?"},
            {}, self.args())
        self.assertEqual(len(problems), 1)


class TestBuildAxis(unittest.TestCase):
    def test_travel_limits_come_from_the_swept_range(self):
        """Never 0-4095: outside the swept range the mapping is extrapolation
        and the load guard has never been there."""
        fit = TestAcceptance.good_fit()
        axis = fit_sysid.build_axis(1, fit, {}, measured=True)
        self.assertEqual(axis.min_counts, 1898)
        self.assertEqual(axis.max_counts, 2198)
        self.assertNotEqual(axis.max_counts, 4095)

    def test_dynamics_are_carried_through(self):
        fit = TestAcceptance.good_fit(backlash_rad=0.007, deadband_counts=6.3)
        steps = {"step_latency_s": 0.021, "rise_time_s": 0.11,
                 "max_rate_rad_s": 1.4}
        axis = fit_sysid.build_axis(1, fit, steps, measured=True)
        self.assertAlmostEqual(axis.backlash_rad, 0.007)
        self.assertAlmostEqual(axis.step_latency_s, 0.021)
        self.assertAlmostEqual(axis.max_rate_rad_s, 1.4)

    def test_unmeasured_axis_still_refuses_to_convert(self):
        """The contract's central safety property, asserted end to end."""
        from contract.servo_contract import CalibrationError

        axis = fit_sysid.build_axis(1, TestAcceptance.good_fit(), {},
                                    measured=False)
        with self.assertRaises(CalibrationError):
            axis.angle_to_counts(0.0)

    def test_measured_axis_round_trips_angle_and_counts(self):
        axis = fit_sysid.build_axis(1, TestAcceptance.good_fit(), {},
                                    measured=True)
        self.assertEqual(axis.angle_to_counts(0.0), 2048)
        for angle_deg in (-3.0, -1.0, 1.0, 3.0):
            angle = math.radians(angle_deg)
            recovered = axis.counts_to_angle(
                axis.center_counts + axis.sign * axis.counts_per_rad * angle)
            self.assertAlmostEqual(recovered, angle, places=9)


if __name__ == "__main__":
    unittest.main()
