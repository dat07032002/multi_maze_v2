# Simulation

MuJoCo model of the printed 256 x 226 mm maze. Milestone M0 of the plan: the
board geometry and a ball-on-plate model trustworthy enough to build on.

```bash
python -m sim.mjcf_builder --out board.xml   # compile, print geom counts
python -m sim.render_maze                    # four PNGs into artifacts/sim_preview
python -m sim.view_maze                      # interactive; arrows tilt, R resets
python -m sim.view_maze --flat               # bare plate, no walls or holes
python -m pytest test/test_physics.py test/test_mjcf_builder.py -q
```

`python`, not `python3`: the simulator runs on the Windows development machine,
where `python3` is a Microsoft Store alias that is not installed. The rest of
the repo says `python3` because the hardware tools run on the Linux box with the
servos and camera attached.

| File | What |
| --- | --- |
| `parameters.json` | every constant, each marked `measured` / `derived` / `design` / `assumed` |
| `mjcf_builder.py` | `maze_256x226.json` -> MJCF |
| `board_state.py` | frame conversions, rate-limited tilt driver, restitution measurement |
| `actuator.py` | commanded angle -> realised plate angle, on the measured chain |
| `analytic_model.py` | closed-form double integrator, used off the simulator |
| `route.py` | arc-length resampled route, projection, lookahead, clearance |
| `rollout.py` | closed loop: measure -> filter -> predict -> control -> act |
| `run_baseline.py` | the analytic baseline's score, and a track overlay |
| `render_maze.py` | offscreen review renders |
| `view_maze.py` | interactive viewer |

## What the model contains

847 geoms: 795 floor boxes with the 15 holes cut out, 42 walls, 4 tag pads, 4
frame rails, a catch tray and the ball. Roughly 46 000 physics steps/s on one
core, so a 20 Hz control step costs about 1 ms and a million env steps is 18
minutes single-core.

Three decisions worth knowing before changing anything:

**Holes are real gaps, not a scripted termination.** MuJoCo has no boolean
subtraction, and a mesh with holes cut in it collides as its convex hull -- a
solid slab. So the floor is decomposed into y-bands with each hole's x-interval
subtracted. Band edges are placed at equal steps in *half-width* rather than
uniformly in y, because a circle's half-width slope runs away at its top and
bottom: uniform 2 mm -> 0.25 mm strips cost 6.8x the boxes and improved the rim
only from 4.50 mm to 1.92 mm, while the current scheme holds 0.419 mm against a
0.5 mm budget.

**The cut errs outward, never inward.** A band is removed wherever any part of
it falls inside the circle, so no sliver of floor pokes into a hole for the ball
to catch on. Measured intrusion is 0.

**Tilt is applied with a rate limit, and sets `qvel` as well as `qpos`.** Writing
`qpos` alone leaves the contact solver seeing a stationary plate. Commanding a
step change is worse: an 8 degree jump in one millisecond launches the ball
about 10 mm into the air with no contacts at all. `TiltDriver` slews at the
measured 6.9 deg/s roll and 8.2 deg/s pitch. The full actuator model -- dead
time, backlash, quantisation -- arrives in M1.

## The actuator model (M1)

`request -> tilt.py compensator -> counts -> [dead time -> lag and rate limit ->
linkage backlash] -> plate`. The compensator is imported from
`tag_vision.control.tilt`, not reimplemented, so sim and hardware cannot drift.

Against the rig:

| | modelled | measured | |
| --- | ---: | ---: | ---: |
| roll latency (5 %) | 184.0 ms | 184.9 ms | −0.5 % |
| roll rise (10-90 %) | 256.0 ms | 256.1 ms | −0.0 % |
| pitch latency (5 %) | 150.0 ms | 150.3 ms | −0.2 % |
| pitch rise (10-90 %) | 229.0 ms | 229.6 ms | −0.3 % |
| roll error, compensation off | 0.880° | 0.876° | +0.5 % |
| roll error, compensation on | 0.280° | 0.261° | +7.3 % |
| pitch error, compensation off | 0.622° | 0.427° | +45.6 % |

Three things worth knowing before changing it:

**Backlash goes last, after the lag.** The slack is in the linkage, so the shaft
moves smoothly and the plate picks up only once the slack is taken up. That is
what makes a reversal cost ~277 counts of servo travel before the plate moves at
all -- and makes a reversal *smaller* than 1.35° move it not at all. Both are
pinned by tests, because they are precisely what the action-rate penalty is
priced against.

**`step_latency_s` is not the dead time.** sysid defines latency as the 5 %
crossing, so a first-order response spends `tau·ln(1/0.95)` of it just climbing
to 5 %. Using 0.185 s as pure dead time would put the modelled crossing 3.2 %
late for a knowable reason. The pure dead times are 0.17896 s and 0.14495 s.

**Quantisation is one count plus a deadband, not a 40-count grid.** The servo
takes any integer count; 40 counts is the smallest reliable *change*. What is
real is the measured deadband -- 6.4 counts roll, 27.9 pitch.

**The slew ceiling is the one consequential unknown left.** `max_rate` is
`max()` of the sample-to-sample rate during the step runs, so it is a floor on
capability, not a ceiling, and biased high besides. sysid only stepped
0.195-0.584°, where a rate limit changes the response by **0 ms**; a policy
commands up to 8° swings, where it is worth **275 ms on a 4° step** — comparable
to the whole dead time. `actuator.slew_limit_scale` (1.0 = clamp at the observed
rate, 5.0 = effectively unclamped) exists so this is randomised at M4 rather than
asserted. One 10-minute rig run settles it:
`tools/sysid_actuator.py --step-counts 40 80 120 400 800`.

## The analytic layer and the baseline (M2)

Between contacts each axis is an independent double integrator,
`x'' = 7.007·sin(beta)`, plus a lumped linear damping. MuJoCo agrees with it to
**0.138 mm over the 250 ms prediction horizon** — 0.5 mm at 500 ms — which is
what the predictor, the Kalman filter and the baseline all rely on.

```bash
python -m sim.run_baseline --seeds 10 --render artifacts/baseline
```

| | |
| --- | ---: |
| success | **10/10** |
| fell in a hole | 0/10 |
| route completion | 99.4 % mean, 99.3 % worst |
| cross-track | 2.84 mm mean, 11.44 mm worst |
| time to goal | 53.0 s (budget 60 s) |

**This is the bar RL has to clear at M3 and M4.** Two things about it matter as
much as the success rate. The working region is narrow — 45 mm/s drops to 8/10,
cross-track gain 4.0 to 7/10 — because a fixed law has very little margin
against 240 ms of loop delay, which is most of the argument for learning one.
And 53 s of a 60 s budget is not much room: the baseline is slow because it has
to be.

**`ball.linear_damping` was discovered here, not assumed.** Fitting
`dv/dt = A·sin(beta) − c·v` to the simulator recovers `A = 7.0555` against the
analytic 7.0071 — the gain is right to 0.7 % — and `c = 0.202 /s`. Without that
term the analytic model overshoots MuJoCo by 7.8 % at 0.4° and 2.2 % at 4°,
worst when the ball is slow, which is exactly where a controller is trying to
hold it still. It is derived from the *sim*, so M5 must re-fit it from a real
coast-down.

## Known limits

- **Wall impacts penetrate ~0.22 mm** at 0.30 m/s, against 0.0025 mm for rolling
  contact. Larger than the 0.1 mm the M0 plan asked for; it is a property of the
  contact solver at a 1 ms timestep and is bounded by a test at 0.3 mm.
- **Restitution is emergent, not declared.** MuJoCo takes a solref damping
  ratio, so `ball.wall_restitution` is a target realised through
  `sim.wall_dampratio`. Read the achieved value with
  `board_state.measure_restitution`; M5 fits the damping ratio to a real bounce.
- **Pitch backlash is modelled flat at 1.35°, and it is position-dependent.**
  Real pitch runs ~1.5° mid-range against 0.4-0.6° near the extremes, so over the
  ±2° validation sequence the model reads 0.622° where the rig reads 0.427° —
  an effective 0.91° out there. The flat figure is what `PROJECT_STATUS.md`
  mandates for simulation, and erring high is the safe direction: a policy
  trained against more lost motion than the rig has is over-prepared, one
  trained against less arrives expecting authority that is not there.
  Tabulating the position dependence is on the M5 list.
- **Four parameters are still assumed**, all flagged in `parameters.json`:
  rolling resistance and wall restitution (both dominant, both fitted at M5),
  floor sliding friction and ball density (both low-consequence -- the ball
  never slips, and `a = (5/7) g sin(theta)` is mass-independent).
- The rolling-resistance model costs about 3 % of the gravity drive at 4
  degrees. Physics is validated against `(5/7) g sin(theta)` with rolling
  resistance switched off, where it agrees to 0.8 %.
