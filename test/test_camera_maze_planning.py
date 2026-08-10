import cv2
import numpy as np

from tag_vision.planning.grid_planner import plan_route
from tag_vision.planning.maze_map import CameraMazeMap, build_camera_maze_map


def test_camera_segmentation_finds_yellow_wall_and_hole_not_corner_tag():
    pixels_per_mm = 2.0
    width_mm, height_mm = 120.0, 80.0
    image = np.full((160, 240, 3), 235, dtype=np.uint8)
    # One yellow wall, one physical black hole, and one tag-like corner blob.
    cv2.rectangle(image, (115, 0), (125, 159), (0, 210, 255), -1)
    cv2.circle(image, (70, 80), 15, (10, 10, 10), -1)
    cv2.circle(image, (12, 148), 15, (10, 10, 10), -1)

    maze = build_camera_maze_map(
        image, board_width_mm=width_mm, board_height_mm=height_mm,
        pixels_per_mm=pixels_per_mm, resolution_mm=1.0,
        ball_radius_mm=5.5, safety_margin_mm=1.5)

    assert len(maze.hole_centres_mm) == 1
    assert np.allclose(maze.hole_centres_mm[0], (35.0, 39.5), atol=0.6)
    assert maze.wall_mask[40, 60]
    assert maze.hole_mask[40, 35]


def _synthetic_map() -> CameraMazeMap:
    height, width = 80, 120
    occupied = np.zeros((height, width), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    occupied[:, 58:63] = True
    occupied[34:47, 58:63] = False
    clearance = cv2.distanceTransform(
        (~occupied).astype(np.uint8), cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE)
    return CameraMazeMap(
        topdown_bgr=np.zeros((height, width, 3), np.uint8),
        wall_mask=occupied.copy(), hole_mask=np.zeros_like(occupied),
        raw_occupied=occupied.copy(), occupied=occupied,
        clearance_mm=clearance, resolution_mm=1.0,
        board_width_mm=120.0, board_height_mm=80.0,
        hole_centres_mm=[])


def test_coordinate_round_trip_and_route_avoids_obstacles():
    maze = _synthetic_map()
    point = (23.0, 61.0)
    assert np.allclose(maze.grid_to_board(maze.board_to_grid(point)), point)

    route = plan_route(maze, (15.0, 60.0), (105.0, 20.0))

    assert len(route.points_mm) > 2
    assert route.length_mm > 90.0
    assert route.minimum_clearance_mm > 0.0
    assert all(not maze.occupied[maze.board_to_grid(point)]
               for point in route.points_mm)
