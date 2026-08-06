# Multi Maze V2: board vision, maze design, and tilt servos

A ball-maze robot built from scratch, standalone Python, no ROS.

Three parts exist today:

- **Vision** -- a calibrated overhead camera and four AprilTags give the board's
  two tilt angles; a hybrid blue-colour/circle detector locates a 5.5 mm-radius
  marble in metric board coordinates.
- **Maze design** -- generators, a rescaler, an STL exporter, and validators for
  the printed 256 x 226 mm insert. See [`maze_design/README.md`](maze_design/README.md).
- **Servos** -- a from-scratch FEETECH STS3215 bus driver and command tools.

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

## Repository layout

```text
calib/                         camera, board, zero, and printable tag files
contract/                      vendored simulator/policy coordinate contract
docs/PROJECT_STATUS.md         work completed, evidence, limits, next steps
tag_vision/core/board_geometry.py
tag_vision/core/board_pose.py
tag_vision/core/ball_detection.py
tools/manual_tilt_angle.py     standalone live viewer
tools/record_ball_dataset.py   standalone data recorder
tools/*.py                     calibration, tag, and ROS diagnostic utilities
test/                          synthetic geometry, pose, and detector tests
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

See [the project status](docs/PROJECT_STATUS.md) for detailed measurements and
the remaining validation plan.
