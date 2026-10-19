"""Digitise the 2022 Codeforces rating-heatmap screenshot (competitive-programming/.github/readme/cf-heatmap.png)
into src/data/tiles/cp_heatmap.json for the cf-heatmap tile.

Run: python scripts/tile-data/cp_heatmap.py [path/to/cf-heatmap.png]   (needs pillow + numpy)

The screenshot is the repo's own documentation example (no handle shown). Cells are read at their grid
centres and matched to the userscript's tier colours (cf-heatmap/cf-rating-heatmap.user.js, composited on white).
The README's tooltip (rgba(0,0,0,.8)) covers ~60 cells, but they show through at 20%, so under the tooltip
the same palette is matched after scaling by 0.2: nothing is guessed.

Output: {"year": 2022, "cells": "<365 chars, one per day from Jan 1>"} where '.' = no solve and '0'..'9' = tier index
(0 = 3000+ dark red ... 9 = below 1200 grey), plus the colour count per tier.
"""
import json, os, sys
import numpy as np
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Desktop/kenyi/projects/competitive-programming/.github/readme/cf-heatmap.png")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/data/tiles/cp_heatmap.json")

# userscript TIERS (rgba, alpha .9) composited on white, highest tier first
TIERS = [(170, 0, 0), (255, 0, 0), (255, 100, 100), (255, 187, 85), (255, 204, 136),
         (255, 85, 255), (170, 170, 255), (119, 221, 187), (119, 255, 119), (204, 204, 204)]
ON_WHITE = [t if i == 1 else tuple(round(0.9 * v + 25.5) for v in t) for i, t in enumerate(TIERS)]  # 2600 is opaque rgb(255,0,0)
EMPTY = (232, 232, 232)
CANDS = [EMPTY] + ON_WHITE  # index 0 = empty, 1..10 = tier 0..9

im = np.array(Image.open(SRC).convert("RGB")).astype(int)
TOOLTIP = (791, 255, 1175, 498)  # x0, y0, x1, y1 in the screenshot
cx = lambda c: 171 + (c - 1) * 30.627  # column centres (col 0 holds only Jan 1, a Saturday)
cy = lambda r: 217.5 + r * 30.67  # row 0 = Sunday


def nearest(rgb, scale=1.0):
    d = [sum((a - scale * b) ** 2 for a, b in zip(rgb, c)) for c in CANDS]
    return int(np.argmin(d)), min(d) ** 0.5


def classify(c, r):
    x, y = int(round(cx(c))), int(round(cy(r)))
    ys, xs = np.mgrid[y - 11:y + 12, x - 11:x + 12]
    px = im[y - 11:y + 12, x - 11:x + 12].reshape(-1, 3)
    inside = ((xs >= TOOLTIP[0]) & (xs < TOOLTIP[2]) & (ys >= TOOLTIP[1]) & (ys < TOOLTIP[3])).reshape(-1)
    if inside.mean() < 0.5:  # mostly visible: palette on white, outside-tooltip pixels only
        p = px[~inside]
        return nearest(np.median(p, axis=0), 1.0), False
    p = px[inside]
    p = p[p.max(axis=1) <= 75]  # drop the tooltip's text pixels
    return nearest(np.median(p, axis=0), 0.2), True


cells, worst, decoded = [], 0, 0
for d in range(365):  # Jan 1 2022 is a Saturday
    c, r = (d + 6) // 7, (d + 6) % 7
    (k, err), via_tooltip = classify(c, r)
    worst = max(worst, err * (5 if via_tooltip else 1))
    decoded += via_tooltip
    cells.append("." if k == 0 else str(k - 1))
s = "".join(cells)
assert s[142] == "1", "2022-05-23 (the README's tooltip day, hardest problem 2600) must be the 2600 tier"
assert worst < 12, f"colour match too loose: {worst}"
json.dump({"year": 2022, "cells": s}, open(OUT, "w"))
print(f"{sum(ch != '.' for ch in s)} coloured days of 365, {decoded} cells read through the tooltip, worst match {worst:.1f}")
print({t: s.count(str(t)) for t in range(10)})
