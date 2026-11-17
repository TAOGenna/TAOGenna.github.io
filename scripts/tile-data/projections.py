"""Spreads for the Projections tile: real captures of issues Nº 00 and Nº 01, from the repo's README.

Each file is a full magazine spread (16:10). The tile shows them as an open magazine whose right page
turns to the next spread on hover. Written at 1120x700, about 2x the book's size in a 2x2 tile.

Run: python scripts/tile-data/projections.py [path/to/projections]   (needs pillow)
Source: <projections>/docs/readme/*.jpg (1600x1000), default ~/Desktop/kenyi/projects/projections
"""
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/kenyi/projects/projections")
SRC = os.path.join(REPO, "docs/readme")
PUB = os.path.join(HERE, "../../public/projects/tiles")

# Turn order in the tile (index 0 is the still frame).
SPREADS = [
    "n01-cover",
    "n01-contents",
    "n00-jpeg",
    "n00-muon",
    "n00-matmul",
    "n01-winding-numbers",
    "n01-self-compressing",
    "n01-crawler",
]
W, H = 1120, 700

for i, name in enumerate(SPREADS):
    im = Image.open(f"{SRC}/{name}.jpg").convert("RGB").resize((W, H), Image.LANCZOS)
    out = f"{PUB}/proj_{i}_{name}.jpg"
    im.save(out, quality=80, optimize=True, progressive=True)
    print(out, os.path.getsize(out) // 1024, "KB")
