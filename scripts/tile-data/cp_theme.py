"""Crops for the cf-theme tile: the same region of the repo's two real screenshots (light / dark theme).

Region = Codeforces logo, nav menu with its pastel bars, contest title and subtitle (no
standings table at all, so no other users' handles). Written ~2x the reel card (2x2 tile) image area.

Run: python scripts/tile-data/cp_theme.py   (needs pillow)
Source: ../../competitive-programming/.github/readme/cf-theme-{light,dark}.png (2560x1280)
"""
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/Desktop/kenyi/projects/competitive-programming/.github/readme")
PUB = os.path.join(HERE, "../../public/projects/tiles")
BOX = (70, 30, 1560, 655)  # 1490x625: stops under the contest subtitle, above the standings header
W = 1300

for name in ("light", "dark"):
    im = Image.open(f"{SRC}/cf-theme-{name}.png").convert("RGB").crop(BOX)
    im = im.resize((W, round(W * im.height / im.width)), Image.LANCZOS)
    im.save(f"{PUB}/cp_theme_{name}.jpg", quality=82)
    print(name, im.size, "top-left", "#%02x%02x%02x" % im.getpixel((1, 1)))
