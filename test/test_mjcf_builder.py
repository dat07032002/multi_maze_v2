"""M0 acceptance: does the compiled model match the layout it came from?

Checked against the *compiled* MuJoCo geoms rather than the intermediate
rectangle lists, so a mistake in the emitter is caught as well as one in the
decomposition. The layout is the source of truth; nothing here re-derives the
maze, it only asks whether what compiled is what was asked for.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from sim.mjcf_builder import (build_mjcf, flat_board_layout, floor_rects,
                              layout_to_model, load_layout, load_parameters,
                              model_to_layout, tag_pad_rects, wall_rects)


@pytest.fixture(scope="module")
def layout():
    return load_layout()


@pytest.fixture(scope="module")
def params():
    return load_parameters()


@pytest.fixture(scope="module")
def model(layout, params):
    return mujoco.MjModel.from_xml_string(build_mjcf(layout, params))


def geom_rects(model, layout, prefix):
    """Compiled geoms whose name starts with ``prefix``, back in layout coords."""
    rects = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if not name.startswith(prefix):
            continue
        px, py = model.geom_pos[i][:2]
        sx, sy = model.geom_size[i][:2]
        x, y = model_to_layout(px, py, layout)
        rects.append((x - sx, y - sy, x + sx, y + sy))
    return rects


# -- walls -------------------------------------------------------------------
def assert_same_rects(got, expected, tol=2e-6):
    """Compare two rectangle sets by nearest match, not by sorted order.

    The MJCF writes positions at %.6f, so corners differ in the seventh decimal
    -- enough to reorder a lexicographic sort and pair unrelated rectangles
    against each other, which reads as a 50 mm error in a model that is right.
    """
    got = np.asarray(list(map(tuple, got)), dtype=float)
    expected = np.asarray(list(map(tuple, expected)), dtype=float)
    assert got.shape == expected.shape, (
        f"got {len(got)} rectangles, expected {len(expected)}")

    unmatched = list(range(len(got)))
    worst = 0.0
    for target in expected:
        distances = [float(np.max(np.abs(got[i] - target))) for i in unmatched]
        best = int(np.argmin(distances))
        worst = max(worst, distances[best])
        unmatched.pop(best)
    assert not unmatched
    assert worst < tol, f"worst corner mismatch {worst * 1000:.6f} mm"


def test_every_wall_run_compiles_to_a_geom_in_the_right_place(model, layout):
    assert_same_rects(geom_rects(model, layout, "wall_"), wall_rects(layout))


def test_pad_comb_runs_are_dropped(layout):
    """Pads are stored as ~13 parallel wall runs per corner, not as blocks.

    Emitting them as walls would work by accident -- at 2 mm spacing and 3 mm
    thickness they overlap into a solid block -- but it buries 58 redundant
    geoms in the model and hides the fact that the layout has a pad concept.
    """
    kept = len(wall_rects(layout))
    total = len(layout["walls_h"]) + len(layout["walls_v"])
    assert total - kept == 58, f"dropped {total - kept} runs, expected 58"

    pads = tag_pad_rects(layout)
    for x0, y0, x1, y1 in wall_rects(layout):
        mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        for px0, py0, px1, py1 in pads:
            assert not (px0 < mid[0] < px1 and py0 < mid[1] < py1), (
                f"wall at {mid} survived inside a tag pad")


def test_walls_are_clamped_to_the_board(model, layout):
    W, H = layout["board_width"], layout["board_height"]
    for x0, y0, x1, y1 in geom_rects(model, layout, "wall_"):
        assert -1e-9 <= x0 and x1 <= W + 1e-9
        assert -1e-9 <= y0 and y1 <= H + 1e-9


# -- pads and frame ----------------------------------------------------------
def test_four_solid_pads_at_the_layout_footprints(model, layout):
    got = geom_rects(model, layout, "pad_")
    assert len(got) == 4
    assert_same_rects(got, tag_pad_rects(layout))


def test_outer_frame_closes_all_four_sides(model, layout):
    """The layout has none: border_walls_printed is false because the physical
    picture frame retains the ball. Without this the ball leaves the world."""
    W, H = layout["board_width"], layout["board_height"]
    rects = geom_rects(model, layout, "frame_")
    assert len(rects) == 4

    def spans(pred):
        return any(pred(*r) for r in rects)

    assert spans(lambda x0, y0, x1, y1: x1 <= 1e-9 and y0 <= 0 and y1 >= H)
    assert spans(lambda x0, y0, x1, y1: x0 >= W - 1e-9 and y0 <= 0 and y1 >= H)
    assert spans(lambda x0, y0, x1, y1: y1 <= 1e-9 and x0 <= 0 and x1 >= W)
    assert spans(lambda x0, y0, x1, y1: y0 >= H - 1e-9 and x0 <= 0 and x1 >= W)


# -- floor and holes ---------------------------------------------------------
def test_no_floor_ever_intrudes_into_a_hole(layout, params):
    """Must be exactly zero. A sliver of floor poking into a hole is something
    the ball can catch on, and it would make the hole quietly smaller than the
    7.5 mm the reward function assumes."""
    rects = floor_rects(layout, params["sim.floor_rim_step"])
    worst = 0.0
    for x0, y0, x1, y1 in rects:
        for (hx, hy), r in zip(layout["holes"], layout["hole_radii"]):
            near_x = min(max(hx, x0), x1)
            near_y = min(max(hy, y0), y1)
            gap = r - math.hypot(near_x - hx, near_y - hy)
            worst = max(worst, gap)
    assert worst < 1e-12, f"floor intrudes {worst * 1000:.6f} mm into a hole"


def test_over_removal_stays_within_the_rim_step_budget(layout, params):
    """Cutting a circle out of axis-aligned boxes has to err one way; it errs
    toward removing too much, and this bounds how much."""
    budget = params["sim.floor_rim_step"]
    rects = floor_rects(layout, budget)
    W, H = layout["board_width"], layout["board_height"]
    holes = [(np.array(h), r) for h, r
             in zip(layout["holes"], layout["hole_radii"])]

    worst = 0.0
    for y in np.arange(0.00025, H, 0.00025):
        covering = [(x0, x1) for x0, y0, x1, y1 in rects if y0 <= y <= y1]
        for x in np.arange(0.00025, W, 0.00025):
            if any(x0 <= x <= x1 for x0, x1 in covering):
                continue
            gap = min(float(np.hypot(x - c[0], y - c[1])) - r for c, r in holes)
            worst = max(worst, gap)
    assert worst <= budget, (
        f"floor missing up to {worst * 1000:.3f} mm outside a hole edge, "
        f"budget {budget * 1000:.3f} mm")


def test_hole_count_and_radii_survive_the_translation(layout):
    assert len(layout["holes"]) == 15
    assert set(layout["hole_radii"]) == {0.0075}


# -- frames ------------------------------------------------------------------
def test_layout_and_model_coordinates_round_trip(layout):
    for x, y in [(0.0, 0.0), (0.256, 0.226), (0.128, 0.113), (0.0593, 0.2134)]:
        mx, my = layout_to_model(x, y, layout)
        back = model_to_layout(mx, my, layout)
        assert back == pytest.approx((x, y))
    # Board centre maps to the hinge origin, which is what makes the two tilt
    # axes pass through the middle of the plate.
    assert layout_to_model(0.128, 0.113, layout) == pytest.approx((0.0, 0.0))


def test_flat_board_has_no_walls_holes_or_pads():
    flat = flat_board_layout()
    assert flat["walls_h"] == [] and flat["walls_v"] == []
    assert flat["holes"] == [] and flat["tag_pads"] == []
    model = mujoco.MjModel.from_xml_string(build_mjcf(flat))
    assert geom_rects(model, flat, "wall_") == []
    assert geom_rects(model, flat, "pad_") == []
    assert len(geom_rects(model, flat, "frame_")) == 4


def test_ball_uses_the_authoritative_radius_not_the_design_margin(model, params):
    """maze_256x226.json says ball_radius 0.006. That is corridor planning
    margin; the marble is 0.0055 and the detector is tuned to it."""
    radius = float(model.geom("ball").size[0])
    assert radius == pytest.approx(0.0055)
    assert radius != pytest.approx(load_layout()["ball_radius"])
    assert float(model.body("ball").mass[0]) == pytest.approx(
        params["ball.mass"], rel=1e-3)
