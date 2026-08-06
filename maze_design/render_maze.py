"""Render a maze design top-down, with the route and hole roles marked."""
import json
import math
import sys

from PIL import Image, ImageDraw

SP = __file__.rsplit("/", 1)[0] + "/"
path = sys.argv[1] if len(sys.argv) > 1 else SP + "maze_v1.json"
out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".json", ".png")

# Derive the roles file from the layout being rendered. This used to be a fixed
# maze_v1 path regardless of argv, so rendering any other maze silently coloured
# its holes by another maze's roles.
import os

roles = {}
for candidate in (path.replace(".json", "_roles.json"),
                  path.replace(".json", "_hole_roles.json")):
    if os.path.isfile(candidate):
        roles = json.load(open(candidate))
        break

layout = json.load(open(path))
W, H = layout["board_width"], layout["board_height"]
T = layout["wall_thickness"]
SCALE = 3400
MARGIN = 46
width_px = int(W * SCALE) + 2 * MARGIN
height_px = int(H * SCALE) + 2 * MARGIN

image = Image.new("RGB", (width_px, height_px), (250, 250, 248))
draw = ImageDraw.Draw(image)


def px(x, y):
    return (MARGIN + x * SCALE, height_px - MARGIN - y * SCALE)


# board
a, b = px(0, H), px(W, 0)
draw.rectangle([a[0], a[1], b[0], b[1]], fill=(246, 232, 200),
               outline=(120, 92, 48), width=3)

# Tag pads come from the layout when present. The old hardcoded 23 mm centred
# grid predates the edge-to-edge grid and drew them in the wrong place.
COLS, ROWS = 11, 9
if layout.get("tag_pads"):
    pads = [((p["centre_mm"][0] - p["pad_mm"][0] / 2) / 1000.0,
             (p["centre_mm"][1] - p["pad_mm"][1] / 2) / 1000.0,
             p["pad_mm"][0] / 1000.0, p["pad_mm"][1] / 1000.0,
             p.get("pocket_mm"), p["centre_mm"]) for p in layout["tag_pads"]]
else:
    pitch_x, pitch_y = W / COLS, H / ROWS
    pads = [(ci * pitch_x, cj * pitch_y, pitch_x, pitch_y, None,
             ((ci + 0.5) * pitch_x * 1000, (cj + 0.5) * pitch_y * 1000))
            for ci, cj in ((0, 0), (COLS - 1, 0), (COLS - 1, ROWS - 1), (0, ROWS - 1))]

def draw_tag_pads():
    """Drawn after the walls: the maze emits walls on every side of a tag cell,
    so pads painted first were completely covered by them."""
    for x_lo, y_lo, pad_w, pad_h, pocket, centre_mm in pads:
        p0 = px(x_lo, y_lo + pad_h)
        p1 = px(x_lo + pad_w, y_lo)
        draw.rectangle([p0[0], p0[1], p1[0], p1[1]],
                       fill=(226, 232, 244), outline=(105, 130, 175), width=2)
        if pocket:
            cx, cy = centre_mm[0] / 1000.0, centre_mm[1] / 1000.0
            q0 = px(cx - pocket[0] / 2000.0, cy + pocket[1] / 2000.0)
            q1 = px(cx + pocket[0] / 2000.0, cy - pocket[1] / 2000.0)
            draw.rectangle([q0[0], q0[1], q1[0], q1[1]],
                           fill=(255, 255, 255), outline=(60, 88, 140), width=3)
        draw.text(((p0[0] + p1[0]) / 2 - 22, (p0[1] + p1[1]) / 2 - 6),
                  "tag pocket", fill=(60, 88, 140))

half = T * SCALE / 2.0
for x_lo, x_hi, y in layout["walls_h"]:
    p0, p1 = px(min(x_lo, x_hi), y), px(max(x_lo, x_hi), y)
    draw.rectangle([p0[0], p0[1] - half, p1[0], p1[1] + half],
                   fill=(96, 66, 30))
for y_lo, y_hi, x in layout["walls_v"]:
    p0, p1 = px(x, min(y_lo, y_hi)), px(x, max(y_lo, y_hi))
    draw.rectangle([p0[0] - half, p1[1], p1[0] + half, p0[1]],
                   fill=(96, 66, 30))

draw_tag_pads()

# route
route = [px(p[0], p[1]) for p in layout["waypoints"]]
draw.line(route, fill=(0, 168, 255), width=7, joint="curve")

# holes, coloured by role when known
role_by_index = {}
if roles:
    role_by_index = {int(k): v for k, v in roles.items()}
for index, ((hx, hy), r) in enumerate(zip(layout["holes"], layout["hole_radii"])):
    centre = px(hx, hy)
    radius = r * SCALE
    role = role_by_index.get(index, "block")
    fill = (30, 30, 34) if role == "block" else (150, 30, 30)
    draw.ellipse([centre[0] - radius, centre[1] - radius,
                  centre[0] + radius, centre[1] + radius],
                 fill=fill, outline=(0, 0, 0))

start = px(*layout["start_planned"])
goal = px(*layout["goal_planned"])
draw.ellipse([start[0] - 15, start[1] - 15, start[0] + 15, start[1] + 15],
             outline=(0, 170, 60), width=6)
draw.ellipse([goal[0] - 15, goal[1] - 15, goal[0] + 15, goal[1] + 15],
             outline=(220, 40, 40), width=6)
draw.text((start[0] + 20, start[1] - 8), "START", fill=(0, 140, 50))
draw.text((goal[0] - 62, goal[1] - 8), "GOAL", fill=(200, 30, 30))

length = sum(math.dist(layout["waypoints"][k], layout["waypoints"][k + 1])
             for k in range(len(layout["waypoints"]) - 1))
draw.text((MARGIN, 14),
          f"{W*1000:.0f} x {H*1000:.0f} mm   "
          f"{len(layout['holes'])} holes (15 mm)   "
          f"walls {T*1000:.0f} mm thick x {layout['wall_height']*1000:.0f} mm tall   "
          f"corridors 20 mm   route {length*1000:.0f} mm",
          fill=(40, 40, 40))
image.save(out)
print("wrote", out, image.size)
