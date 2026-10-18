"""Prepare the Photonics tile's eye diagrams from the post's own OptiSystem figures.

Run: python scripts/tile-data/photonics_eyes.py   (needs pillow + numpy)
Crops each eye diagram to its plot interior (drops axis ticks/labels) and maps luminance to the
site palette (ink -> steel -> lavender -> white). The traces are the post's real simulation output.
"""
import os

import numpy as np
from PIL import Image

POST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/content/post")
PUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../public/projects/tiles")

# name -> (source figure, crop box in source pixels)
EYES = {
    "ph_eye_single": ("ph_eye_diagram_single.png", (58, 62, 640, 618)),  # single channel
    "ph_eye_ch1": ("ph_3c3.png", (70, 62, 630, 632)),  # 1553.32 nm, 36 GHz
    "ph_eye_ch2": ("ph_2c3.png", (30, 35, 664, 658)),  # 1552.92 nm, 28 GHz
    "ph_eye_ch3": ("ph_1c3.png", (58, 35, 630, 655)),  # 1552.52 nm, 20 GHz
}
STOPS = [(0.0, (11, 15, 20)), (0.18, (22, 30, 40)), (0.5, (143, 169, 184)), (0.82, (207, 210, 245)), (1.0, (255, 255, 255))]


def ramp(t):
    out = np.zeros(t.shape + (3,))
    for (a, ca), (b, cb) in zip(STOPS, STOPS[1:]):
        m = (t >= a) & (t <= b)
        k = ((t - a) / (b - a))[m][:, None]
        out[m] = np.array(ca) * (1 - k) + np.array(cb) * k
    return out


for name, (src, box) in EYES.items():
    im = Image.open(os.path.join(POST, src)).convert("RGB").crop(box)
    a = np.asarray(im, dtype=float) / 255
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]) ** 0.7
    out = Image.fromarray(ramp(np.clip(lum, 0, 1)).astype(np.uint8))
    out = out.resize((380, round(380 * out.height / out.width)), Image.LANCZOS)
    out.save(os.path.join(PUB, name + ".jpg"), quality=86)
    print(name, out.size)
