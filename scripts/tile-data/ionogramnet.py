"""Prepare the IonogramNET tile's images from the repo's real network outputs.

Each assets/descarga (k).png is a pair: the (binarised) input ionogram and the CNN's soft prediction.
We crop the two plot rectangles (no titles/axes) and recolor them into the site palette:
  raw  : echo/noise pixels -> ink on a pale panel
  pred : probability -> pale -> lavender -> ink ramp
Run: python scripts/tile-data/ionogramnet.py   (needs pillow + numpy)
"""
import os

import numpy as np
from PIL import Image

SRC = os.path.expanduser("~/Desktop/kenyi/projects/ionospheric-echo-detection-with-convolutional-neural-networks/assets/")
PUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../public/projects/tiles")
hexrgb = lambda h: np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], float)
BG, INK, LAV = hexrgb("#f7f9fa"), hexrgb("#0b0f14"), hexrgb("#cfd2f5")
RAW_BOX, PRED_BOX = (14, 34, 318, 409), (435, 34, 741, 409)  # plot rectangles minus the frame line; both resized to W x H below

for k in range(1, 5):
    im = Image.open(SRC + f"descarga ({k}).png")
    flat = Image.new("RGB", im.size, (255, 255, 255))
    flat.paste(im, mask=im.split()[3])
    g = np.asarray(flat.convert("L"), float) / 255

    v = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).crop(RAW_BOX), float) / 255
    raw = BG * (1 - v[..., None]) + INK * v[..., None]

    p = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).crop(PRED_BOX).resize((304, 375)), float) / 255
    p = np.clip(p / np.percentile(p, 99.7), 0, 1)
    lo = np.clip(p / 0.5, 0, 1)[..., None]  # bg -> lavender over 0..0.5
    hi = np.clip((p - 0.5) / 0.5, 0, 1)[..., None]  # lavender -> ink over 0.5..1
    pred = (BG * (1 - lo) + LAV * lo) * (1 - hi) + INK * hi

    Image.fromarray(raw.astype(np.uint8)).save(PUB + f"/ion_{k}_raw.png", optimize=True)
    Image.fromarray(pred.astype(np.uint8)).save(PUB + f"/ion_{k}_pred.png", optimize=True)
    print(k, "ok")
