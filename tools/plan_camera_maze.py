#!/usr/bin/env python3
"""Map the physical maze from the camera and plan a clearance-aware route.

Examples:
    python3 tools/plan_camera_maze.py --click
    python3 tools/plan_camera_maze.py --start-mm 30 200 --goal-mm 220 25
    python3 tools/plan_camera_maze.py --auto-endpoints --no-window
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.core.ball_detection import BlueBallDetector  # noqa: E402
from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.core.board_pose import BoardPoseEstimator  # noqa: E402
from tag_vision.planning.grid_planner import nearest_free, plan_route  # noqa: E402
from tag_vision.planning.maze_map import (  # noqa: E402
    build_camera_maze_map,
    rectify_board,
)
from tools.camera_imu_check import camera_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def auto_endpoints(maze) -> tuple[tuple[float, float], tuple[float, float]]:
    count, labels = cv2.connectedComponents((~maze.occupied).astype(np.uint8), 8)
    if count <= 1:
        raise ValueError("camera map contains no connected free space")
    sizes = np.bincount(labels.ravel())
    label = int(np.argmax(sizes[1:]) + 1)
    cells = np.argwhere(labels == label)
    # Restrict endpoint selection toward corridor centres; otherwise the
    # farthest pair tends to sit one pixel from an inflated obstacle.
    clear = maze.clearance_mm[cells[:, 0], cells[:, 1]]
    candidates = cells[clear >= np.percentile(clear, 65)]
    if len(candidates) < 2:
        candidates = cells
    seed = candidates[int(np.argmax(
        maze.clearance_mm[candidates[:, 0], candidates[:, 1]]))]
    first = candidates[int(np.argmax(np.sum((candidates - seed) ** 2, axis=1)))]
    second = candidates[int(np.argmax(np.sum((candidates - first) ** 2, axis=1)))]
    return maze.grid_to_board(first), maze.grid_to_board(second)


def clicked_points(maze, existing_start=None):
    image = maze.topdown_bgr
    header_px = 44
    board_width_mm = maze.board_width_mm
    board_height_mm = maze.board_height_mm
    points = [] if existing_start is None else [tuple(existing_start)]
    prompt = ["click START then GOAL" if existing_start is None else
              "click GOAL in the reachable region"]
    _count, component = cv2.connectedComponents(
        (~maze.occupied).astype(np.uint8), 8)

    def snapped_free(point):
        cell = maze.board_to_grid(point)
        cell = nearest_free(maze.occupied, cell)
        return maze.grid_to_board(cell), cell

    if points:
        points[0], _ = snapped_free(points[0])

    def point_from_pixel(x, y):
        height, width = image.shape[:2]
        return (x / (width - 1) * board_width_mm,
                (height - 1 - y) / (height - 1) * board_height_mm)

    def callback(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 2:
            return
        if y < header_px:
            prompt[0] = "click on the maze below this instruction bar"
            return
        point = point_from_pixel(x, y - header_px)
        cell = maze.board_to_grid(point)
        if maze.occupied[cell]:
            prompt[0] = "inside RED ball-clearance zone - click free floor"
            return
        if points:
            _, start_cell = snapped_free(points[0])
            if component[cell] != component[start_cell]:
                prompt[0] = "unreachable from START - click the clear region"
                return
        points.append(point)
        prompt[0] = "click GOAL in the reachable region"

    cv2.namedWindow("camera maze | choose route")
    cv2.setMouseCallback("camera maze | choose route", callback)
    while len(points) < 2:
        maze_view = image.copy()
        high_occupied = cv2.resize(
            maze.occupied.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST).astype(bool)
        tint = maze_view.copy()
        tint[high_occupied] = (30, 30, 210)
        maze_view = cv2.addWeighted(maze_view, 0.70, tint, 0.30, 0)
        if points:
            _, start_cell = snapped_free(points[0])
            reachable = component == component[start_cell]
            unreachable = (~maze.occupied) & (~reachable)
            high_unreachable = cv2.resize(
                unreachable.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(bool)
            maze_view[high_unreachable] = (
                maze_view[high_unreachable] * 0.45).astype(np.uint8)
        display = np.full(
            (image.shape[0] + header_px, image.shape[1], 3), 245,
            dtype=np.uint8)
        display[header_px:] = maze_view
        cv2.putText(display, prompt[0], (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        for point in points:
            px = int(point[0] / board_width_mm * (image.shape[1] - 1))
            py = header_px + int(
                (board_height_mm - point[1]) / board_height_mm
                * (image.shape[0] - 1))
            cv2.circle(display, (px, py), 7, (0, 255, 0), -1)
        cv2.imshow("camera maze | choose route", display)
        if cv2.waitKey(20) & 0xFF in (27, ord("q")):
            raise KeyboardInterrupt
    cv2.destroyWindow("camera maze | choose route")
    return points[0], points[1]


def route_overlay(maze, route, start, goal) -> np.ndarray:
    image = maze.topdown_bgr.copy()
    height, width = image.shape[:2]
    occupied = cv2.resize(
        maze.occupied.astype(np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST).astype(bool)
    tint = image.copy()
    tint[occupied] = (40, 40, 190)
    image = cv2.addWeighted(image, 0.72, tint, 0.28, 0)

    def pixel(point):
        return (int(round(point[0] / maze.board_width_mm * (width - 1))),
                int(round((maze.board_height_mm - point[1]) /
                          maze.board_height_mm * (height - 1))))

    route_pixels = np.asarray([pixel(point) for point in route.points_mm],
                              dtype=np.int32)
    if len(route_pixels) >= 2:
        cv2.polylines(image, [route_pixels], False, (255, 220, 0), 3,
                      cv2.LINE_AA)
    cv2.circle(image, pixel(start), 7, (0, 190, 0), -1)
    cv2.circle(image, pixel(goal), 7, (0, 0, 230), -1)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--image", type=Path,
                        help="map a saved native camera frame instead of live video")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--calib", type=Path,
                        default=ROOT / "calib/camera_calib.json")
    parser.add_argument("--board", type=Path,
                        default=ROOT / "calib/board_tags.json")
    parser.add_argument("--start-mm", type=float, nargs=2)
    parser.add_argument("--goal-mm", type=float, nargs=2)
    parser.add_argument("--click", action="store_true")
    parser.add_argument("--auto-endpoints", action="store_true")
    parser.add_argument("--pixels-per-mm", type=float, default=2.0)
    parser.add_argument("--resolution-mm", type=float, default=1.0)
    parser.add_argument("--ball-radius-mm", type=float, default=5.5)
    parser.add_argument(
        "--safety-margin-mm", type=float, default=0.5,
        help="clearance beyond the 5.5 mm ball radius (default: 0.5 mm)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    geometry = BoardGeometry.load(args.board)
    estimator = BoardPoseEstimator(args.calib, geometry, min_tags=4)
    detector = BlueBallDetector(
        estimator, radius_m=args.ball_radius_mm / 1000.0)
    topdowns = []
    latest_frame = latest_pose = None

    if args.image:
        native = cv2.imread(str(args.image))
        frames = [] if native is None else [native]
        if not frames:
            print(f"Could not read {args.image}")
            return 1
    else:
        source = camera_source(args.camera)
        capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        if not capture.isOpened():
            print(f"Could not open camera {source!r}")
            return 1
        # Let exposure/white balance settle before collecting the median map.
        for _ in range(5):
            capture.read()
        frames = []
        for _ in range(args.frames):
            ok, native = capture.read()
            if ok:
                frames.append(native)
        capture.release()

    reprojections = []
    for native in frames:
        gray = cv2.cvtColor(native, cv2.COLOR_BGR2GRAY)
        pose = estimator.estimate(gray)
        if pose is None or pose.reprojection_px > 4.0:
            continue
        topdowns.append(rectify_board(
            native, estimator, pose, pixels_per_mm=args.pixels_per_mm))
        latest_frame, latest_pose = native, pose
        reprojections.append(pose.reprojection_px)
    if not topdowns:
        print("No four-tag board pose; cannot make a metric map")
        return 1
    topdown = np.median(np.stack(topdowns), axis=0).astype(np.uint8)
    maze = build_camera_maze_map(
        topdown,
        board_width_mm=geometry.board_width_m * 1000.0,
        board_height_mm=geometry.board_height_m * 1000.0,
        pixels_per_mm=args.pixels_per_mm,
        resolution_mm=args.resolution_mm,
        ball_radius_mm=args.ball_radius_mm,
        safety_margin_mm=args.safety_margin_mm,
    )

    ball = None
    if latest_frame is not None and latest_pose is not None:
        calibrated = cv2.resize(
            latest_frame, estimator.image_size, interpolation=cv2.INTER_AREA)
        ball = detector.detect(calibrated, latest_pose)
    start = tuple(args.start_mm) if args.start_mm else (
        tuple(ball.board_xy_m * 1000.0) if ball is not None else None)
    goal = tuple(args.goal_mm) if args.goal_mm else None
    if args.auto_endpoints and (start is None or goal is None):
        automatic = auto_endpoints(maze)
        start = start or automatic[0]
        goal = goal or automatic[1]
    if args.click and (start is None or goal is None):
        start, goal = clicked_points(maze, start)

    route = None
    route_error = None
    if start is not None and goal is not None:
        try:
            route = plan_route(maze, start, goal)
        except ValueError as exc:
            route_error = str(exc)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "artifacts/camera_maze" / stamp
    output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(output / "topdown.png"), topdown)
    cv2.imwrite(str(output / "walls.png"), maze.wall_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "holes.png"), maze.hole_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "occupied_inflated.png"),
                maze.occupied.astype(np.uint8) * 255)
    metadata = {
        "board_size_mm": [maze.board_width_mm, maze.board_height_mm],
        "resolution_mm": maze.resolution_mm,
        "ball_radius_mm": args.ball_radius_mm,
        "safety_margin_mm": args.safety_margin_mm,
        "valid_mapping_frames": len(topdowns),
        "median_reprojection_px": float(np.median(reprojections)),
        "hole_centres_mm": maze.hole_centres_mm,
        "detected_ball_mm": ((ball.board_xy_m * 1000.0).tolist()
                             if ball else None),
        "start_mm": start,
        "goal_mm": goal,
        "route_error": route_error,
    }
    if route is not None:
        overlay = route_overlay(maze, route, start, goal)
        cv2.imwrite(str(output / "route_overlay.png"), overlay)
        metadata["route"] = {
            "points_mm": route.points_mm,
            "length_mm": route.length_mm,
            "minimum_clearance_mm": route.minimum_clearance_mm,
        }
    else:
        overlay = topdown
    (output / "map.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"map: {output}")
    print(f"poses: {len(topdowns)}/{len(frames)}, median reprojection "
          f"{np.median(reprojections):.2f} px")
    print(f"holes detected: {len(maze.hole_centres_mm)}")
    print(f"occupied after inflation: {np.mean(maze.occupied):.1%}")
    if ball is not None:
        print(f"ball: {ball.board_xy_m[0]*1000:.1f}, "
              f"{ball.board_xy_m[1]*1000:.1f} mm")
    if route is not None:
        print(f"route: {route.length_mm:.0f} mm, {len(route.points_mm)} points, "
              f"minimum extra clearance {route.minimum_clearance_mm:.1f} mm")
    else:
        if route_error:
            print(f"route not generated: {route_error}")
            print("Choose endpoints in the same clear region, or inspect "
                  "occupied_inflated.png before reducing the safety margin.")
        else:
            print("map only: supply --start-mm/--goal-mm or --click to plan")
    if not args.no_window:
        cv2.imshow("camera-derived maze route", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 2 if route_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
