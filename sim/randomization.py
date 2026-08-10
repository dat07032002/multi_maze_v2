"""Domain randomisation, kept deliberately narrow.

The documented failure of DR is not under-randomising, it is over-randomising:
too wide and the policy learns something robust and slow, and on a 1023 mm route
with a 60 s budget slow may simply not finish. The M2 baseline already needs
53 s of that 60 at 40 mm/s, so there is very little room to buy robustness with
caution.

This rig is measured to a standard that makes narrowness affordable. Everything
in the actuator is measured; what is genuinely unknown is a short list, and each
entry below says which it is.

The ranges come from ``parameters.json`` -- ``assumed`` entries carry their own
``range`` -- rather than being restated here, so widening a range is a change to
the parameter file and shows up in review as one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.mjcf_builder import DEFAULT_PARAMETERS

#: What gets randomised, and why it is on the list.
RANDOMISED = {
    "ball.rolling_friction_length":
        "dominant unknown: sets coast-down and terminal speed. Fitted at M5.",
    "ball.wall_restitution":
        "dominant unknown: decides whether a bounce crosses a 20 mm corridor. "
        "Realised through sim.wall_dampratio, so that is what moves.",
    "ball.linear_damping":
        "derived from the simulator at M2, not from the rig. Until a real "
        "coast-down exists it is a fitted guess.",
    "camera.latency":
        "NOT MEASURED. There is no camera timestamping in the repo at all, and "
        "it adds directly to the 185 ms actuator dead time.",
    "camera.position_noise":
        "never measured on labelled data; the 98 % figure in PROJECT_STATUS is "
        "a replay recovery rate.",
    "camera.dropout_rate": "never measured.",
    "actuator.slew_limit_scale":
        "the one consequential unknown left in the actuator: sysid only ever "
        "stepped 0.195-0.584 deg, where a rate limit is invisible, and a policy "
        "commands up to 8 deg swings where it is worth 275 ms.",
    "actuator.centre_bias":
        "absolute level repeats to only about 0.4 deg, so this is a per-episode "
        "bias rather than per-step noise.",
    "ball.floor_sliding_friction":
        "wide range is harmless: pure rolling needs mu >= 0.042 at the hard "
        "limit and every plausible value is far above that.",
    "ball.density":
        "wide range is harmless: a = (5/7) g sin(theta) is mass-independent.",
}

# The ranges in parameters.json are uncertainty/stress bounds, not a claim that
# every endpoint is equally likely on the assembled rig.  The MPC teacher data
# was generated over 0..25 % of those ranges, and the full-maze sensitivity
# sweep showed that even the upper quarter contains qualitatively different
# plants when several uncertainties combine. Keep scale=1 available for stress
# testing. Until M5 replaces assumptions with measurements, ordinary training
# starts with this narrow profile and can curriculum-ramp toward 0.25.
REALISTIC_DR_SCALE = 0.10
TEACHER_MAX_DR_SCALE = 0.25
FULL_STRESS_SCALE = 1.0


class Randomizer:
    """Samples a parameter set per episode from the ranges in parameters.json."""

    def __init__(self, path: str | Path = DEFAULT_PARAMETERS,
                 scale: float = REALISTIC_DR_SCALE,
                 enabled: bool = True):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.enabled = enabled
        #: ``scale`` shrinks every range toward its nominal value. 0 is
        #: equivalent to disabled and 1 is the full documented range; anything
        #: above 1 is refusing the argument above and is not offered.
        self.scale = float(np.clip(scale, 0.0, 1.0))
        self.spec = {}
        for name in RANDOMISED:
            entry = raw.get(name)
            if entry is None:
                continue
            self.spec[name] = (entry["value"], entry.get("range"))

    def sample(self, params: dict, rng: np.random.Generator) -> dict:
        out = dict(params)
        if not self.enabled or self.scale == 0.0:
            return out

        for name, (nominal, bounds) in self.spec.items():
            if name == "actuator.centre_bias":
                # Per-episode level offset, symmetric about the measured value.
                spread = params["actuator.level_repeatability"] * self.scale
                out[name] = [float(v + rng.normal(0.0, spread / 2.0))
                             for v in nominal]
                continue
            if bounds is None:
                continue
            lo, hi = bounds
            lo = nominal + (lo - nominal) * self.scale
            hi = nominal + (hi - nominal) * self.scale
            out[name] = float(rng.uniform(lo, hi))

        # Restitution is not directly settable; it is realised through the wall
        # damping ratio, so that is the knob that has to move with it.
        if "ball.wall_restitution" in self.spec:
            target = out["ball.wall_restitution"]
            nominal = self.spec["ball.wall_restitution"][0]
            ratio = params["sim.wall_dampratio"] * (nominal / max(target, 1e-3))
            out["sim.wall_dampratio"] = float(np.clip(ratio, 0.05, 2.0))

        return out

    def describe(self) -> str:
        lines = [f"domain randomisation, scale {self.scale:g}"]
        for name, (nominal, bounds) in sorted(self.spec.items()):
            lines.append(f"  {name:<34} {nominal}  range {bounds}")
        return "\n".join(lines)


#: Stable flattened ordering of the randomised parameters. ``actuator.centre_bias``
#: is a per-axis pair, so it occupies two slots; everything else is scalar.
#: Fixed here rather than derived per call, because it is the column order of a
#: recorded array that has to keep meaning across runs.
PHYSICS_VECTOR_NAMES = tuple(
    name for entry in sorted(RANDOMISED)
    for name in ((f"{entry}.roll", f"{entry}.pitch")
                 if entry == "actuator.centre_bias" else (entry,)))


def physics_vector(params: dict) -> np.ndarray:
    """The sampled plant, as a fixed-order vector.

    This is what a privileged teacher conditions on and what an adaptation
    module regresses onto: in simulation the values are known exactly, on the
    rig they are not, and the whole point of estimating them online is that the
    two situations differ only here. Recorded per transition alongside the
    demonstrations so the pairing survives into training.
    """
    values = []
    for entry in sorted(RANDOMISED):
        value = params[entry]
        if entry == "actuator.centre_bias":
            values.extend(float(component) for component in value)
        else:
            values.append(float(value))
    return np.asarray(values, dtype=np.float32)
