# Vision project status

Last updated: 2026-08-06

> **Scope note.** This document covers the vision work only, and parts of it
> predate the 2026-08-06 rescale to a 256 x 226 mm board. Numbers measured on
> the 259 x 229 mm board (reprojection, angle noise, detection rates) were not
> re-taken and should be treated as indicative. For maze design see
> [`../maze_design/README.md`](../maze_design/README.md); for the servo stack
> see the repository README.

## Scope

The current milestone is reliable state estimation for a manually tilted maze
board:

```text
camera frame
  -> four-tag board pose
  -> zero-relative alpha/beta
  -> board-constrained blue-marble detection
  -> metric marble centre (x, y)
```

Maze design, reinforcement-learning policy design, training, and deployment are
later phases. The perception output deliberately follows the servo contract's
coordinate conventions so it can feed those phases without changing them.

As of 2026-08-06 a second state-estimation path exists: a BNO086 IMU on an
ESP32, streaming board tilt at 200 Hz. It is not a replacement for the camera --
it gives angle only, not marble position -- but it samples fast enough to
measure actuator dynamics, which the camera at 10-30 Hz does not. See
[Actuator system identification](#actuator-system-identification) below.

## Coordinate and angle conventions

The moving board frame is:

- origin `(0,0,0)` at the physical lower-left board corner when viewed from
  above, adjacent to tag 3;
- `+X` right toward tag 1;
- `+Y` up toward tag 0;
- `+Z` out of the board surface;
- width 256 mm and height 226 mm.

The marble output is the centre `(x,y)` in this frame. Ray/plane intersection
uses `z = 5.5 mm` because the marble radius is 5.5 mm.

Angle order matches the simulator's two nested hinges:

```text
R = Rx(alpha) @ Ry(beta)
alpha = atan2(R[2,1], R[1,1])
beta  = atan2(R[0,2], R[0,0])
```

The result is relative to a rotation captured while the board is level. It is
not an absolute gravity reference.

## Work completed

### Camera calibration

- Built AprilGrid generation, capture, calibration, scaling, and validation
  tools.
- Compared pinhole/radial-tangential and Kannala-Brandt models.
- Selected the Kannala-Brandt calibration: 0.390 px RMS versus 0.808 px for the
  pinhole fit.
- Calibrated at 640 x 400 and uniformly scaled intrinsics to the working
  1280 x 800 image; the camera captures native 1920 x 1200 with the same aspect
  ratio before resizing.
- Added a focus-assist utility and live tag monitor.

### Board geometry and pose

- Defined the board frame and per-tag 3-D corners (259 x 229 mm at the
  time; rescaled to 256 x 226 mm on 2026-08-06).
- Added per-tag in-plane rotation so the installed tag orientations map to the
  correct ArUco corner order.
- Recorded installed layout:

  | ID | Location | Rotation |
  | --- | --- | ---: |
  | 0 | top-left | 0° |
  | 1 | bottom-right | -90° |
  | 2 | top-right | 0° |
  | 3 | bottom-left | 180° |

- Added geometry validation, known-ID filtering, fisheye point undistortion,
  iterative PnP, reprojection RMS, simulator-compatible angle extraction, and
  saved zero rotation.
- Benchmarked corner refinement on the live rig. `CORNER_REFINE_CONTOUR` gave
  approximately 1.84 px median reprojection and 0.016°/0.006° static alpha/beta
  noise. `SUBPIX` gave about 2.38 px and 0.09°/0.07°; AprilTag refinement did
  not provide complete four-tag poses on this setup.
- Added a standalone camera viewer so missing ROS (`rclpy`) no longer blocks
  angle measurement.
- Added full-frame undistortion and a board boundary/origin/+X/+Y overlay.

### Blue-marble detection

- Added a board-projected region of interest with distortion-aware sampled
  edges.
- Excluded expanded, known tag regions to suppress blue-cast black borders.
- Tuned broad and strict HSV/BGR blue masks for the installed camera.
- Added morphology, contour/ellipse fitting, circularity, expected-size, and
  strict-colour scoring.
- Added pose-aware pixel-to-board conversion using a ray intersection with the
  marble-centre plane.
- Added confidence rejection, temporal proximity scoring, a 65 mm maximum
  step, EMA output smoothing, and stale-track reset.
- Added Hough-circle recovery for a ball merged into a rail, wall, or blue
  handling tool.
- Added a recorder that saves frames, masks, per-frame measurements, manifests,
  and summaries.

### Actuator system identification

Built and completed on hardware 2026-08-06. `calib/servo_calibration.json` has
`measured=True`; `angle_to_counts` no longer raises.

The chain is: BNO086 IMU on an ESP32 streaming board tilt at 200 Hz, a sweep
tool that drives the servos and records settled angles, and a fit tool that
writes the contract only when the measurement supports the model it encodes.

#### Results

Calibrated from a conditioned sweep over ±1200 counts, centred on level.

| | roll (servo 1 → alpha) | pitch (servo 2 → beta) |
| --- | ---: | ---: |
| deg/count | 0.00487 | 0.00502 |
| counts_per_rad | 11776 | 11412 |
| centre (level) | 2215 | 2053 |
| calibrated range | 929–3329 | 902–3302 |
| span measured | 11.49° | 12.22° |
| linearity residual | 1.99% (0.229°) | 1.68% (0.205°) |
| **backlash, conditioned** | 0.031° | 0.140° |
| **backlash, unconditioned** | **1.06° mean, 1.42° max** | **0.94° mean, 1.54° max** |
| step latency | 185 ms | 150 ms |
| rise time (10–90%) | 256 ms | 230 ms |
| max rate | 6.9°/s | 8.2°/s |
| `max_tilt` | ±4.0° (headroom ±5.42°) | ±4.0° (headroom ±5.78°) |

A straight line holds across the **whole** working range: 12° of span at under
2% residual, better relative accuracy than the earlier ±320-count sweep managed
over 3°. The nonlinear `AxisCalibration` that seemed inevitable is not needed.

##### Backlash is the dominant error, and only one number is operational

| error source | magnitude |
| --- | ---: |
| IMU noise | 0.006° |
| command resolution (40 counts) | 0.19° |
| linear model residual | 0.23° |
| **backlash, unconditioned** | **~1.35°** |

Backlash is roughly **seven times everything else combined**. The conditioned
figures (0.031°/0.140°) describe what the mechanism can do when every approach
comes from the same side; the unconditioned ones describe what a controller that
reverses direction actually gets. **Use 1.35° for the simulator.** Putting the
conditioned number in would make a simulated roll axis forty times more precise
than the hardware.

The hysteresis is flat at 1.3–1.4° across the middle of the sweep and falls to
zero at both turnarounds, which is the signature of constant lost motion rather
than load-dependent compliance -- compliance would grow toward the extremes
where gravity torque is highest, and it does the opposite. So it models as a
reversal-triggered deadband, and needs no dependence on tilt angle.

It is also feedforward-correctable, and the constant is now measured: inject
about **277 counts on roll, 269 on pitch** at a direction reversal. With a ~100 ms
irreducible dead time capping any feedback loop at 0.2-0.3 Hz, that is the
highest-value control code for this rig.

#### Travel limits, 2026-08-06 (superseded by the renumbering below)

Stepping outward from level until the servo ran out of counts:

| | measured extremes | tilt span | safe range (300-count backoff) | safe span |
| --- | --- | ---: | --- | ---: |
| roll | 75 – 4025 | 18.06° | 371 – 3730 | 15.55° |
| pitch | 51 – 4088 | 20.96° | 347 – 3794 | 17.87° |

All four directions ended on the **servo's 0/4095 count limit**, not on a stall
or a gain collapse. That does not prove the plate was still free: the guard
cannot tell "resting against a stop" from "travelling" once the servo stops
being asked to move, so a 300-count backoff is applied on each side and recorded
in `calib/servo_travel_limits.json`.

Within the safe range:

| | usable about current level | best if level were centred |
| --- | ---: | ---: |
| roll | ±5.41° | **±7.77°** |
| pitch | ±8.12° | **±8.93°** |

**±10° is not reachable on either axis** inside the safe range. Roll caps at
±7.77° even perfectly centred, and would need roughly 29% more linkage ratio;
pitch needs about 12%. Level is currently 2.36° off-centre in roll's span and
0.81° in pitch's, so re-centring is worth more than any other single change.

These limits are deliberately **not** the same as `min_counts`/`max_counts` in
`servo_calibration.json`. Those are the range the linear fit was measured over
(±320 counts); these are the range it is safe to drive. Commanding angles across
the full safe range would extrapolate the fit roughly five times past its data.

#### Method: the two things that made it work

**Approach conditioning.** Every sweep point is reached from a fixed side, three
times, before the angle is read (`--condition-counts`, `--condition-cycles`).
Backlash fell from 0.734°/0.197° to 0.024°/0.040° — thirty-fold on roll — and
reproduced across two runs. Returning repeatedly to one count from a fixed side
settles to ±0.009° after about three cycles, against 0.73° when direction
varies. The lost motion is real but entirely deterministic; earlier sweeps
sampled both sides of it and read the difference as unreliability.

**Configuration pinned.** `STS3215Bus.CANONICAL_CONFIG` holds the motion
settings every measurement was taken under, with `apply_config` (write and
verify) and `check_config` (report drift). `ramp_to` no longer writes
`ACCELERATION` or `GOAL_SPEED` unless asked; it previously rewrote both on every
call, so any caller could invalidate the calibration silently.

`ACCELERATION = 50` is measured, not chosen: a 160-count step gave 270 ms of
board rise at the old value of 20 and 146 ms at 50, with 150 no better. Goal
speed was measured *not* to limit — rise time was flat across 100/200/500/1000.
**`GOAL_SPEED = 0` is not "maximum"** on this firmware; under it a 160-count step
moved the board 0.031° in 1.26 s, and `ramp_to` now rejects it.

#### What was ruled out

Servo 1 looked unreliable for most of the session — 0.73° of apparent backlash,
6.7% gain reproducibility against servo 2's 0.3%. Four explanations were tested
and failed, and recording them matters because each was believed at the time:

| Hypothesis | Test | Result |
| --- | --- | --- |
| Plate mechanically constrained | load guard aborts | guard was too strict, and the loads it tripped on were a decoding bug (see below) |
| Loose linkage | physical inspection | tight |
| Servo droop under load | encoder vs goal at P = 32/64/128 | lands on goal ±3 counts, zero load at rest; raising P changed nothing |
| IMU drift | 6-minute still log | 0.013° drift, 0.126° range — the sensor is good |

The answer was approach direction, above. Two corrections worth keeping:

- The load guard originally aborted on **load alone**, which rejected every
  upward move and read convincingly as a seized mechanism. A stall is high load
  **and no progress**; `ramp_to` now requires both, over several samples, and
  judges progress from an optional `motion_probe` (the IMU) rather than the
  encoder — servo 1 once advanced 200 counts through a dead zone while the plate
  moved 0.08°.
- Servo 1's travel limit was **numeric, not mechanical**. Level sat at count
  4093 of 4095, so the controller could not be handed a larger number.
  `calibrate_middle` (FEETECH's own middle-position calibration) renumbered both
  servos to 2048 with the board moving 0.0007°. An earlier version of this
  document recommended re-indexing the horn mechanically; that was wrong, and it
  was wrong because the mechanism was assumed rather than read — the servo's own
  limit registers said plainly the constraint was numeric.

#### Acceptance: why the linearity gate was changed

Both axes sit at 1.3–2.4% of span across runs, straddling the original 2% limit,
so pass/fail flipped between runs and between servos. Percentage-of-span is a
relative measure, and over a ~3° span it condemns errors nothing can act on:

| | model error | one 40-count command | share |
| --- | ---: | ---: | ---: |
| roll | 0.057° | 0.190° | 30% |
| pitch | 0.071° | 0.175° | 41% |

The linear model is finer than the smallest command this rig can issue — 40
counts, below which steps do not reliably keep their sign. `fit_sysid` now
accepts a residual under half of one commandable step, records the comparison in
its notes, and still blocks anything larger; two tests cover both directions.
`RATIO_MIN` was also widened from 0.05 to 0.01, because this rig's genuine ~20:1
reduction (ratio 0.050) tripped a guessed plausibility bound.

#### Limits to carry into the simulator

- **Gain depends on where it is measured.** Servo 2 read 0.00540 deg/count over
  ±400 counts, 0.00463 over ±320 centred at 2047, and 0.00437 over ±320 centred
  at 2018. The stored value is an average over the swept range and **must not be
  extrapolated past it**. The falloff is shared by both axes and is real
  geometry; a nonlinear mapping is the eventual right answer.
- **Action quantisation belongs in the model.** Commands under 20 counts do not
  reliably keep their sign; 40 counts (~0.19°) is the practical floor. A policy
  trained on continuous tilt will learn authority the hardware does not have.
- **Backlash is deterministic, not noise, and it dominates.** Model it as
  reversal-triggered lost motion, not as Gaussian error, and use the
  **unconditioned** figure of ~1.35° -- roughly seven times every other error
  source combined. The conditioned 0.031°/0.140° describes calibration
  conditions, not operation.
- **Dynamics are conditional on the pinned config.** Latency and rise time were
  measured at `ACCELERATION = 50`, `GOAL_SPEED = 500`, recorded in
  `servo_config`. A controller that changes them must re-measure.
- **~100 ms of the latency is electrical and irreducible**, which caps any
  feedback loop's bandwidth at roughly 0.2–0.3 Hz. Feedforward through the
  calibration plus reversal compensation is the right control shape here; a fast
  PID is not.

#### Range after renumbering, 2026-08-06

`calibrate_middle` renumbered both servos so level reads 2048, moving the count
wrap away from roll's travel. The board moved 0.003° -- nothing physical
changed, only the labels -- and roll's usable range improved:

| | before | after |
| --- | ---: | ---: |
| roll | ±5.40° | **±7.00°** |
| pitch | ±8.39° | ±8.36° |

Loads at every new extreme were 24-28, peak 40 anywhere, so the plate is not
contacting anything: the limit was purely the count wrap. That evidence was
unreadable before the load-decoding fix.

±10° is still out of reach. Roll would need roughly 30% more linkage ratio, and
the torque case for that is comfortable -- normal operation sits at 3-4% of
rated, so a faster linkage would reach perhaps 5%.

#### Remaining

1. **Camera cross-check.** The only independent validation of the IMU, and the
   one item from the original plan never run — the See3CAM was unplugged.
   Everything above rests on a single sensor whose internal consistency is good
   (0.013° drift, clean axis separation, gains reproducing to 0.3%) but which
   has never been checked against another instrument.
2. Nonlinear `AxisCalibration`, if ±1.4° proves too narrow for the maze.
3. Feedforward + reversal-compensation control layer, mirrored in the simulator.
4. MuJoCo model using the table above.

### Verification

- Added synthetic tests for angle convention, tag corner order and rotation,
  known-tag filtering, pose recovery, zeroing, blue-ball detection, coordinate
  recovery, distractor rejection, outside-board rejection, merged-component
  recovery, full-frame undistortion, and overlay projection consistency.
- Added synthetic tests for the IMU wire format (CRC rejection, resync after a
  dropped byte, sequence-gap counting, sequence wrap, stale-pong rejection) and
  for the sysid fits (gain/sign/centre recovery, backlash recovery, curvature
  showing up in the residual, latency and rise time from a first-order step,
  and the acceptance gate that decides `measured=True`).
- One real defect was found this way: with a step window shorter than the
  settling time, the final value came from a tail that was still moving, so the
  90% threshold was computed against a truncated delta and a confidently wrong,
  too-fast rise time was returned. `_step_metrics` now checks that the tail has
  settled and reports an error instead.
- Current result: **66 tests pass** (21 vision, 18 IMU, 27 sysid).
- A dry render on a real recorded frame placed the undistorted origin at about
  `(288.7, 691.7)` px, with finite +X/+Y endpoints inside the 1280 x 800 view.

## Recorded experiments

### Capture `20260804_172006`

- 335 frames.
- 0 accepted poses at the 4 px limit.
- 123 frames saw all four tags, but reprojection was about 10.83 px.
- Subset inspection isolated the mismatch to the bottom-right tag (ID 1), which
  was physically rotated relative to the configured geometry. The tag layout
  configuration was corrected/aligned before the next capture.

This recording is a useful example of why tag visibility alone is insufficient:
the reprojection gate correctly rejected a geometrically inconsistent solve.

### Capture `20260804_172347`

- 269 frames, recorded with three-tag poses permitted for diagnosis.
- 263 accepted poses (97.77%).
- Tag counts: 166 frames with all four; 94 with `(0,1,3)`; 3 with `(0,2,3)`;
  and 6 with no pose.
- Pose reprojection: 1.895 px median, 1.924 px 90th percentile, 1.963 px max.
- Four-tag static median: alpha 0.0623°, beta 0.8338° with standard deviations
  0.0134°/0.0072°.
- Common `(0,1,3)` median: alpha 0.5326°, beta 1.0298° with standard deviations
  0.0101°/0.0195°.
- The three-tag subset therefore shifted the result about +0.47° alpha and
  +0.20° beta. Four tags remain the production requirement.

The original contour-only detector reported only 10 detections because a blue
pair of scissors touched the marble and merged both objects into one component.
After adding edge-circle recovery, offline replay found a ball in 244 of 249
usable-pose frames (98.0% conditional recovery). Replay processing took about
33 ms/frame including image loading and pose estimation. At 10 Hz, the detected
trajectory had 3.4 mm median, 7.4 mm 90th-percentile, and 29.4 mm maximum
inter-sample movement.

These frames are not manually labeled, and the blue scissors are a major
distractor. The 98.0% number is a replay recovery rate, **not** a ground-truth
precision/recall measurement.

Raw captures remain locally under `artifacts/ball_capture/` and are excluded
from Git because they occupy about 136 MB. The recorder reproduces the same
directory structure.

## Current defaults

| Parameter | Default | Reason |
| --- | ---: | --- |
| Required tags | 4 | avoids observed three-tag angle bias |
| Max reprojection | 4.0 px | rejects inconsistent board geometry |
| Ball radius | 5.5 mm | physical marble radius |
| Ball confidence | 0.54 | tuned against current recordings |
| Ball EMA new-sample weight | 0.45 | reduces coordinate jitter |
| Track reset | after 3 misses | avoids displaying stale coordinates |
| Maximum temporal step | 65 mm | rejects remote blue distractors |

## Limitations and next work

1. Record a freely moving marble without scissors or other blue tools.
2. Label visible/not-visible frames and ball centres to measure precision,
   recall, false positives, coordinate error, jitter, and latency.
3. Cover the centre, rails, corners, shadows, motion blur, several tilt angles,
   and empty-board negative sequences.
4. Consider strict full-board acquisition followed by adaptive local tracking
   only if labeled results expose remaining misses.
5. Measure background colour statistics and seed-and-grow segmentation only if
   fixed thresholds fail under expected lighting variation.
6. Validate alpha/beta against known physical shims; reprojection residual is a
   self-consistency check, not proof of angle accuracy.
7. Re-measure nominal tag coordinates if absolute coordinate/angle errors are
   too large for control.
8. Add a fixed-frame tag or inertial/gravity reference if the camera must move
   without re-zeroing.
9. Begin maze representation and policy design only after the perception
   acceptance criteria are met.

## Suggested perception acceptance criteria

Define these numerically before policy integration. A practical first target is:

- at least 99% four-tag pose availability in the intended operating envelope;
- no accepted poses above the 4 px reprojection gate;
- ball precision and recall each measured on labeled data rather than inferred
  from tracker continuity;
- coordinate error and jitter low enough relative to maze corridor clearance;
- processing latency comfortably below the future control-loop period;
- automatic `not visible` output during genuine occlusion instead of a guessed
  coordinate.
