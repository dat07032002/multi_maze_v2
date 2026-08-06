# Multi Maze V2: board vision, maze design, and tilt servos

A ball-maze robot built from scratch, standalone Python, no ROS.

Four parts exist today:

- **Vision** -- a calibrated overhead camera and four AprilTags give the board's
  two tilt angles; a hybrid blue-colour/circle detector locates a 5.5 mm-radius
  marble in metric board coordinates.
- **Maze design** -- generators, a rescaler, an STL exporter, and validators for
  the printed 256 x 226 mm insert. See [`maze_design/README.md`](maze_design/README.md).
- **Servos** -- a from-scratch FEETECH STS3215 bus driver and command tools.
- **IMU and system identification** -- an ESP32 streams BNO086 tilt at 200 Hz,
  and `tools/sysid_actuator.py` measures the counts-to-angle mapping the servo
  contract needs. See [`firmware/esp32_imu/README.md`](firmware/esp32_imu/README.md).

Not yet built: the simulator, the control loop, and the policy.

The main live application is standalone OpenCV and does **not** require ROS:

```bash
python3 tools/manual_tilt_angle.py
```

Press `z` while the board is level to save the angle reference. Press `q` or
`Esc` to quit.

## What the viewer reports

The window is camera-undistorted and displays:

- `alpha`: rotation about the board X hinge, in degrees;
- `beta`: rotation about the subsequent board Y hinge, in degrees;
- detected tag IDs and pose reprojection error;
- blue-marble centre `(x, y)` in millimetres and detector confidence;
- the magenta board boundary and metric coordinate frame;
- `O (0,0) mm` at the lower-left board corner, near tag 3;
- red `+X` toward the lower-right corner/tag 1;
- green `+Y` toward the upper-left corner/tag 0.

The ball position is its centre projected onto the plane `z = 5.5 mm`, not the
point where it touches the board.

## Current hardware and calibration

| Item | Current value |
| --- | --- |
| Camera | See3CAM_24CUG, auto-selected when available |
| Capture | 1920 x 1200 MJPG, resized uniformly to 1280 x 800 |
| Calibration model | Kannala-Brandt fisheye |
| Calibration RMS | 0.390 px over 88 views / 5108 points |
| Board | 256 x 226 mm |
| Tags | Four 18 mm `tag36h11` tags, IDs 0–3 |
| Marble | Blue, 5.5 mm radius |
| Pose requirement | Four known tags by default |
| Pose rejection | Reprojection RMS greater than 4 px |
| Tilt servos | Two FEETECH STS3215, ids 1 and 2, 1 Mbps on `/dev/ttyUSB0` |
| Servo mode | Position (register 33 = 0), 0-4095 counts over 360 deg |
| IMU | SparkFun BNO086, I2C on ESP32 GPIO16/17, game rotation vector @ 200 Hz |
| IMU link | ESP32-D0WD-V3 on CP2102, 115200; ports resolved by USB id |
| Servo calibration | Measured 2026-08-06. Roll 0.00487 deg/count, pitch 0.00502 |
| Level at | counts 2215 (roll) / 2053 (pitch) |
| Commandable tilt | ±4.0°, calibrated headroom ±5.42° / ±5.78° |
| Mechanical travel | roll ±7.00°, pitch ±8.36° |
| Backlash (operational) | **~1.35°** — the dominant error by ~7x |
| Command resolution | 40 counts (~0.19°) practical floor; under 20 is unusable |

The checked-in tag coordinates in
[`calib/board_tags.json`](calib/board_tags.json) are **derived from the maze
layout's tag pads, not caliper-measured** (`"measured": false`). Tags are glued
to the top of the corner pads, so `z_m = 0.011`: the pad top is 8 mm above the
playing surface and the tag is 3 mm thick. All four shift equally, so alpha and
beta are unaffected, but marble coordinates are -- the ray/plane intersection
depends on that height. Measuring the installed tags remains the route to
absolute accuracy.

The saved zero in [`calib/board_zero.json`](calib/board_zero.json) is specific
to the current camera pose. Re-zero after the camera, board mount, or lens has
moved.

## Installation

Python 3.10 was used during development. On a regular Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

OpenCV must contain `cv2.aruco`; this is why the requirements use the contrib
distribution. The main live viewer accesses the camera directly with OpenCV.
The older calibration/monitoring tools that import `rclpy` additionally require
a sourced ROS 2 installation and `cv_bridge`.

## Everyday commands

Run the combined angle and ball viewer:

```bash
python3 tools/manual_tilt_angle.py
```

Select a camera explicitly or run without a GUI:

```bash
python3 tools/manual_tilt_angle.py --camera 2
python3 tools/manual_tilt_angle.py --zero-on-start --no-window
```

Record a new 500-frame detector dataset at 10 Hz:

```bash
python3 tools/record_ball_dataset.py --frames 500
```

Record even when only three tags are visible (diagnostics only):

```bash
python3 tools/record_ball_dataset.py --frames 500 --min-tags 3
```

Generated frames, masks, JSONL measurements, manifests, and summaries are saved
under `artifacts/ball_capture/<timestamp>/`. This directory is ignored by Git
because raw captures are large.

Bring up the IMU and capture the level reference (do this before any servo
moves, and before anything disturbs the plate):

```bash
python3 tools/imu_monitor.py --seconds 20        # noise and drop check
python3 tools/imu_monitor.py --capture-zero --check-torque-shift
```

Measure the servos against the IMU, then fit the contract:

```bash
python3 tools/sysid_actuator.py --dry-run        # preflight, commands no motion
python3 tools/sysid_actuator.py
python3 tools/fit_sysid.py artifacts/sysid/<stamp>/sysid.json
```

Run all synthetic tests:

```bash
python3 -m pytest -q
```

## How it works

### Board angle

1. Detect known AprilTag IDs in grayscale with OpenCV ArUco.
2. Refine corners with `CORNER_REFINE_CONTOUR`, selected from a live benchmark.
3. Undistort tag pixels with the calibrated Kannala-Brandt camera model.
4. Match pixels to each tag's known board-frame corners.
5. Solve the board-to-camera transform with iterative `solvePnP`.
6. Compare rotation with the saved level reference.
7. Extract angles using the simulator-compatible hinge order
   `R = Rx(alpha) @ Ry(beta)`.

### Marble position

1. Project the board boundary into the image and reject pixels outside it.
2. Exclude expanded tag footprints, which can look blue under the camera's
   fixed white balance.
3. Form broad and strict blue masks using HSV plus BGR channel separation.
4. Clean small mask noise and score contour candidates by strict-blue coverage,
   circularity, and expected projected radius.
5. Use Hough-circle edge recovery when a marble touching a rail or tool merges
   into a larger blue component.
6. Prefer candidates near the previous valid position and reject impossible
   frame-to-frame jumps.
7. Convert the chosen distorted pixel to a calibrated ray and intersect it with
   the board-frame marble-centre plane `z = radius`.
8. Apply an exponential moving average to the reported board `(x, y)`.

Detection remains on the calibrated raw frame. The viewer image and every
overlay are mapped into the same undistorted pixel space, so visualization does
not alter the measurement path.

### Board angle from the IMU

A second, independent path to the same two angles, used as ground truth for
system identification because it samples far faster than the camera.

1. The ESP32 reads the BNO086's game rotation vector at 200 Hz over I2C.
2. Quaternions cross the link as Q14 fixed point in a 17-byte CRC-checked frame.
3. The host converts quaternion to rotation matrix and extracts alpha and beta
   with the **same** `angles_from_rotation` the camera path uses, so the two are
   directly comparable rather than two conventions that happen to agree.
4. Angles are reported relative to a level reference captured in place, using
   `zero.T @ current` -- the convention `BoardPoseEstimator` already uses.

Dropped frames are counted and surfaced, never interpolated over: a missing
report during a step response is a hole in the measurement, not a value to
invent.

### Actuator system identification

`tools/sysid_actuator.py` measures what the servo contract refuses to guess, in
a deliberate order:

1. **Axis discovery** -- move each servo alone and see which angle responds.
   Produces the sign, the axis assignment, and the 2x2 cross-coupling matrix.
   Runs first because nothing else is safe until the signs are known, and
   because significant coupling would invalidate the contract's independent-axis
   shape before any effort is spent fitting it.
2. **Static sweep** -- a load-guarded staircase gives gain, offset, and the
   linearity residual.
3. **Hysteresis** -- the same staircase in reverse gives backlash and deadband.
4. **Step response** -- unramped steps at 200 Hz give latency, rise time, and
   peak rate, with half the measured link round-trip subtracted so the latency
   describes the servo rather than the USB cable.

`tools/fit_sysid.py` writes the contract only when the measurement supports the
linear model it encodes. Failing linearity, plausibility, or coupling checks
produces `measured=False`, which keeps `angle_to_counts` raising -- a contract
that claims to be measured and is not would be worse than one that refuses.

## Repository layout

```text
calib/                         camera, board, zero, and printable tag files
contract/servo_contract.py     policy action -> board angle -> servo counts
docs/PROJECT_STATUS.md         work completed, evidence, limits, next steps
firmware/esp32_imu/            ESP32 sketch streaming BNO086 tilt
tag_vision/core/board_geometry.py
tag_vision/core/board_pose.py
tag_vision/core/ball_detection.py
tag_vision/hardware/sts3215.py FEETECH bus driver
tag_vision/hardware/motion.py  load-guarded point-to-point moves
tag_vision/hardware/imu.py     BNO086 frame reader
tag_vision/control/tilt.py     board angle -> counts, backlash fed forward
tools/manual_tilt_angle.py     standalone live viewer
tools/record_ball_dataset.py   standalone data recorder
tools/imu_monitor.py           IMU bring-up and level-zero capture
tools/sysid_actuator.py        actuator system identification
tools/fit_sysid.py             sysid run -> servo calibration
tools/validate_backlash.py     measures whether compensation actually helps
tools/*.py                     calibration, tag, servo, and ROS utilities
test/                          synthetic geometry, pose, detector, IMU, fit tests
```

## Known limitations

- Four coplanar tags give low noise, but out-of-plane tilt remains more weakly
  conditioned than translation. Known-angle physical validation is still
  required before treating angle values as ground truth.
- Using three tags can keep a pose alive during occlusion, but the tested
  `(0,1,3)` subset shifted alpha/beta by roughly `+0.47°/+0.20°` relative to
  the four-tag solution. The production default therefore requires four tags.
- Blue appearance thresholds were tuned for this camera and lighting. Major
  lighting, exposure, white-balance, board-colour, or marble-colour changes
  require fresh validation.
- The 98% recovery measured during development was an unlabeled replay with a
  blue handling tool present. It demonstrates improved recovery, not a final
  true-positive rate. A labeled, freely moving marble dataset is the next
  validation milestone.
- There is no fixed-frame tag. Moving the camera invalidates the saved level
  reference and requires pressing `z` again.
- **The IMU has never been cross-checked against an independent sensor.** Every
  board-angle number -- gains, backlash, latency -- comes from it alone. Its
  internal consistency is good (0.013° drift over 6 minutes, clean axis
  separation, gains reproducing to 0.3%), but the camera cross-check remains the
  one validation not yet run.
- **The calibration is valid over the range it was swept** (±1200 counts,
  roughly ±5.5°) and must not be extrapolated past it. Within that range a
  straight line holds well -- 12° of span at under 2% residual -- so the
  nonlinear form once thought necessary is not.
- **Absolute level is repeatable to about 0.4°.** The fitted centre moved 85
  counts between two consecutive sweeps with no mechanical change. Relative
  moves are considerably more trustworthy than absolute pose.
- **Commands under 20 counts do not reliably keep their sign.** 40 counts
  (~0.19°) is the practical floor, 80 is repeatable to a few percent. Any
  controller or policy step finer than that is commanding noise.
- **Backlash is the dominant error, at ~1.35° unconditioned.** That is roughly
  seven times the model residual (0.23°), the command resolution (0.19°) and the
  IMU noise (0.006°) combined. It is deterministic, not random: approached
  consistently from one side the axes repeat to 0.03°. Static positioning should
  condition its approach and gets a 40x more precise result; dynamic control
  must feed the lost motion forward instead -- about 277 counts on roll, 269 on
  pitch, injected at a direction reversal. **Simulators must use the
  unconditioned figure**, or the model will be far more precise than the board.
- The IMU level zero is captured **in place**, relying on the board being
  physically level with the IMU already mounted. That removes the need to
  measure a mount offset, but it is perishable: bumping the plate or re-mounting
  the sensor invalidates it, and re-levelling by eye reintroduces exactly the
  offset the approach avoids.
- The straight line in `AxisCalibration` holds within ±1.4°, where the model
  error (0.057°/0.071° rms) is well under one 40-count command. It is not a
  general claim about the linkage, which is a crank and loses authority further
  out; a nonlinear form is the eventual answer if the maze needs more tilt.
- **The ESP32 serial link is unreliable above 115200 and re-delivers frames.**
  Flash reads at 921600 and 460800 both stalled at exactly `0x6000`, and over 5 s
  the port yielded 1298 frames' worth of bytes when the device had sent 1000.
  The reader deduplicates on `(seq, esp_micros)`; telemetry and reflashing stay
  at 115200.

See [the project status](docs/PROJECT_STATUS.md) for detailed measurements and
the remaining validation plan.
