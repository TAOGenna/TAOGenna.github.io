"""Digitise the vrating career chart for the vrating tile (CpGrid / Vrating.astro).

Source: competitive-programming/.github/readme/vrating-career.png, the repo's own screenshot of the
example career of a public Codeforces account (not the site owner's). No handle is kept.
Run: python scripts/tile-data/cp_vrating.py [path/to/vrating-career.png]   (needs pillow + numpy + scipy)

 - the 65 virtual-rating dots sit at equal x steps (one per round), so x = index; each dot's y is the
   centroid of its saturated fill, mapped to a rating with the chart's own gridlines
   (1200 at y=539.2, 400 rating = 58 px)
 - the dashed 'real rating' line is flat after the 7th round; the first seven values are read off the
   6x zoom of the image (the dashes are too sparse to trace by colour)
 - stats are the figures printed on the card.
"""
import json, os, sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Desktop/kenyi/projects/competitive-programming/.github/readme/vrating-career.png")
OUT = os.path.join(HERE, "../../src/data/tiles/cp_vrating.json")

im = np.array(Image.open(SRC).convert("RGB")).astype(float)
mx = im.max(2)

Y1200, PX = 539.2, 0.145  # y of 1200, px per rating point
rating = lambda y: 1200 + (Y1200 - y) / PX
N = 65

# the dot fills are the only saturated pixels (the tier bands are pastel): one blob per round, in x order
sat = (mx - im.min(2)) / np.maximum(mx, 1)
fill = (sat > 0.3) & (mx > 100)
fill[:215] = False
fill[560:] = False
lab, n = ndi.label(fill)
blobs = sorted(ndi.center_of_mass(fill, lab, range(1, n + 1)), key=lambda c: c[1])
assert n == N, f"expected {N} dots, found {n}"
virtual = [round(rating(cy)) for cy, cx in blobs]

real_head = [1850, 1873, 2250, 2212, 2215, 2237, 2622]  # read off the image
real = real_head + [2641] * (N - len(real_head))

# the flat tail of the dashed line really is at 2641: grey pixels near y(2641)
y_real = Y1200 - (2641 - 1200) * PX
band = im[int(y_real) - 2:int(y_real) + 3, 700:990]
assert ((abs(band[..., 0] - band[..., 2]) < 40) & (band[..., 0] < 190)).any(), "dashed tail not where expected"
assert abs(virtual[-1] - 2899) <= 3 and abs(max(virtual) - 3076) <= 3, (virtual[-1], max(virtual))

# Codeforces rating tiers as drawn on the chart: (lo, hi, name)
tiers = [(1100, 1200, "newbie"), (1200, 1400, "pupil"), (1400, 1600, "specialist"), (1600, 1900, "expert"),
         (1900, 2100, "candidate"), (2100, 2400, "master"), (2400, 2600, "intl master"),
         (2600, 3000, "grandmaster"), (3000, 3380, "intl grandmaster")]
ticks = [(0, "2019-07"), (7, "2021-02"), (23, "2022-10"), (60, "2024-01")]  # printed under the x axis

data = {
    "virtual": virtual,
    "real": real,
    "tiers": [{"lo": a, "hi": b, "name": n} for a, b, n in tiers],
    "ticks": [{"i": i, "label": l} for i, l in ticks],
    "range": [1100, 3380],
    "stats": {"virtual": 2899, "delta": 258, "real": 2641, "peak": 3076, "rounds": 65},
}
json.dump(data, open(OUT, "w"))
print("virtual", virtual[:8], "...", virtual[-3:], "peak", max(virtual), "at", virtual.index(max(virtual)))
