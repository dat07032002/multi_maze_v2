"""Regression tests for logical-axis mapping in backlash validation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_backlash import measure_backlash


class ValidateBacklashTest(unittest.TestCase):
    def test_uses_measured_angle_assignment_not_servo_number(self):
        def points(angle, gap):
            key = f"{angle}_rad"
            result = []
            for count in (100, 200, 300):
                result.append({"commanded_counts": count, "direction": "up",
                               key: count / 1000.0})
                result.append({"commanded_counts": count, "direction": "down",
                               key: count / 1000.0 + gap})
            return result

        data = {"experiment_AB": {
            "1": {"angle": "beta", "points": points("beta", 0.1)},
            "2": {"angle": "alpha", "points": points("alpha", 0.2)},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sysid.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            got = measure_backlash(path)
        self.assertAlmostEqual(got["roll"], 0.2)
        self.assertAlmostEqual(got["pitch"], 0.1)


if __name__ == "__main__":
    unittest.main()
