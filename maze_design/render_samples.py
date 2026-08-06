#!/usr/bin/env python3
"""Render every sample_*.json into one contact sheet for picking between."""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = 1500
PAD = 16
COLS = 3


def draw_one(layout, roles):
    W, H = layout["board_width"], layout["board_height"]
    T = layout["wall_thickness"]
    width = int(W * SCALE) + 2 * PAD
    height = int(H * SCALE) + 2 * PAD + 26
    image = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(image)

    def px(x, y):
        return (PAD + x * SCALE, height - PAD - y * SCALE)

    a, b = px(0, H), px(W, 0)
    draw.rectangle([a[0], a[1], b[0], b[1]], fill=(246, 232, 200),
                   outline=(150, 120, 70), width=2)

    COLS_G, ROWS_G, PITCH = 11, 9, 0.023
    gx, gy = (W - COLS_G * PITCH) / 2.0, (H - ROWS_G * PITCH) / 2.0
    for ci, cj in ((0, 0), (COLS_G - 1, 0), (0, ROWS_G - 1),
                   (COLS_G - 1, ROWS_G - 1)):
        p0 = px(gx + ci * PITCH, gy + cj * PITCH + PITCH)
        p1 = px(gx + ci * PITCH + PITCH, gy + cj * PITCH)
        draw.rectangle([p0[0], p0[1], p1[0], p1[1]], fill=(228, 234, 246),
                       outline=(140, 160, 195))

    half = max(1.0, T * SCALE / 2.0)
    for x0, x1, y in layout["walls_h"]:
        p0, p1 = px(min(x0, x1), y), px(max(x0, x1), y)
        draw.rectangle([p0[0], p0[1] - half, p1[0], p1[1] + half],
                       fill=(96, 66, 30))
    for y0, y1, x in layout["walls_v"]:
        p0, p1 = px(x, min(y0, y1)), px(x, max(y0, y1))
        draw.rectangle([p0[0] - half, p1[1], p1[0] + half, p0[1]],
                       fill=(96, 66, 30))

    draw.line([px(p[0], p[1]) for p in layout["waypoints"]],
              fill=(0, 168, 255), width=4, joint="curve")

    for index, ((hx, hy), r) in enumerate(
            zip(layout["holes"], layout["hole_radii"])):
        centre = px(hx, hy)
        radius = r * SCALE
        fill = (150, 30, 30) if roles.get(str(index)) == "dodge" else (28, 28, 32)
        draw.ellipse([centre[0] - radius, centre[1] - radius,
                      centre[0] + radius, centre[1] + radius], fill=fill)

    for point, colour in ((layout["start_planned"], (0, 170, 60)),
                          (layout["goal_planned"], (220, 40, 40))):
        c = px(*point)
        draw.ellipse([c[0] - 8, c[1] - 8, c[0] + 8, c[1] + 8],
                     outline=colour, width=4)
    return image


def main():
    summary = json.load(open(f"{HERE}/samples_summary.json"))
    tiles = []
    for item in summary:
        layout = json.load(open(f"{HERE}/sample_{item['index']}.json"))
        roles = json.load(open(f"{HERE}/sample_{item['index']}_roles.json"))
        tile = draw_one(layout, roles)
        draw = ImageDraw.Draw(tile)
        draw.text((PAD, 6),
                  f"#{item['index']}  route {item['route_mm']:.0f} mm   "
                  f"clearance {item['clearance_mm']:.2f} mm   "
                  f"{item['blocking']} block + {item['dodge']} dodge   "
                  f"middle {item['middle']}/25",
                  fill=(30, 30, 30))
        tiles.append(tile)

    tw, th = tiles[0].size
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * tw, rows * th), (255, 255, 255))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % COLS) * tw, (index // COLS) * th))
    out = f"{HERE}/samples.png"
    sheet.save(out)
    print(f"wrote {out} {sheet.size}")


if __name__ == "__main__":
    main()
