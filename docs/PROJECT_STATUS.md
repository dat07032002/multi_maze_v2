# Vision project status

Last updated: 2026-08-04

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
later phases. The perception output deliberately follows the vendored policy
coordinate contract so it can feed those phases without changing conventions.

## Coordinate and angle conventions

The moving board frame is:

- origin `(0,0,0)` at the physical lower-left board corner when viewed from
  above, adjacent to tag 3;
- `+X` right toward tag 1;
- `+Y` up toward tag 0;
- `+Z` out of the board surface;
- width 259 mm and height 229 mm.

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

- Defined the 259 x 229 mm board frame and per-tag 3-D corners.
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

### Verification

- Added synthetic tests for angle convention, tag corner order and rotation,
  known-tag filtering, pose recovery, zeroing, blue-ball detection, coordinate
  recovery, distractor rejection, outside-board rejection, merged-component
  recovery, full-frame undistortion, and overlay projection consistency.
- Current result: **21 tests pass**.
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
