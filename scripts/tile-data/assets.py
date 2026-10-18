"""Prepare image assets for the Style-transfer and LSeg bento tiles.

- LSeg: for each scene, recolor the repo's matplotlib prediction figure into the
  site palette (cropped to the plot panel, legend/title removed) and size it to
  the photo's aspect (the model ran on a squashed square).
- Style transfer: three content->result pairs + each result's dominant colors.

Run: python scripts/tile-data/assets.py   (needs pillow + numpy + scipy)
"""
import json, os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

P = os.path.expanduser("~/Desktop/kenyi/projects")
HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "../../public/projects/tiles")
DATA = os.path.join(HERE, "../../src/data/tiles")
os.makedirs(PUB, exist_ok=True)

hexrgb = lambda h: tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))
PAPER = hexrgb("#f1efea")

# ---- LSeg ---------------------------------------------------------------
LSEG = P + "/pytorch-language-driven-semantic-segmentation/readme_images/"
# label -> (color in the repo's figure, color in the site palette)
SCENES = [
    ("bear", "bear", [("bear", (196, 84, 100), "#0b0f14"), ("terrain", (212, 220, 148), "#d8ece4")]),
    (
        "people",
        "kids playing",
        [
            ("terrain", (212, 220, 148), "#c9dae2"),
            ("person", (244, 36, 148), "#0b0f14"),
            ("fence", (50, 130, 180), "#cfd2f5"),
            ("vegetation", (12, 124, 68), "#8fa9b8"),
        ],
    ),
    (
        "sign",
        "city scene",
        [
            ("traffic sign", (180, 44, 92), "#0b0f14"),
            ("sky", (164, 92, 28), "#cfd2f5"),
            ("vegetation", (12, 124, 68), "#8fa9b8"),
            ("road", (108, 212, 140), "#c9dae2"),
            ("sidewalk", (124, 156, 116), "#7e8ea1"),
            ("terrain", (212, 220, 148), "#d8ece4"),
        ],
    ),
    # ponytail: 'playing' (tennis) skipped, 'person' and 'wall' share one color in the source figure
]
PX = 240  # mask height in px (shown at ~108 css px)

lseg = []
for key, scene, items in SCENES:
    pred = np.array(Image.open(LSEG + f"{key}_prediction.jpg").convert("RGB")).astype(int)
    d = np.stack([np.abs(pred - np.array(c)).sum(-1) for _, c, _ in items], -1)
    near = d.min(-1) < 70
    # tiny blobs (legend swatches, antialiasing specks) are not region data
    lab, n = ndi.label(near)
    sizes = ndi.sum(near, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 150])
    # plot panel = bbox of the big regions; legend/title/axes fall outside or get filled
    sl = ndi.find_objects(keep.astype(int))[0]
    keep, idx = keep[sl], d.argmin(-1)[sl]
    # fill everything not kept (legend box, text, edges) from the nearest kept pixel
    _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
    idx = idx[iy, ix]
    out = np.zeros(idx.shape + (3,), np.uint8)
    for i, (_, _, dst) in enumerate(items):
        out[idx == i] = hexrgb(dst)
    photo = Image.open(LSEG + f"{key}_original.jpg").convert("RGB")
    ar = photo.width / photo.height
    h = PX
    mask = Image.fromarray(out).resize((round(h * ar), h), Image.LANCZOS)
    photo = photo.resize((round(h * ar), h), Image.LANCZOS)
    mask.save(PUB + f"/lseg_{key}_mask.png")
    photo.save(PUB + f"/lseg_{key}_input.jpg", quality=84)
    lseg.append({"key": key, "scene": scene, "ar": round(ar, 3), "labels": [{"name": n, "color": dst} for n, _, dst in items]})
    print("lseg", key, "panel", out.shape[:2], "ar", round(ar, 3))
json.dump({"scenes": lseg}, open(DATA + "/lseg.json", "w"), indent=1)

# ---- Style transfer -----------------------------------------------------
ST = P + "/pytorch-neural-style-transfer/"
PAIRS = [  # (content, result, object-position)
    ("data/content/green_bridge.jpg", "results/bridge_exp2.jpg", "50% 50%"),
    ("data/content/golden_gate22.jpg", "results/golden2.jpg", "50% 42%"),
    ("data/content/robot.jpg", "results/robot_art1.jpg", "50% 45%"),
]
pairs = []
for i, (c, r, pos) in enumerate(PAIRS):
    Image.open(ST + c).convert("RGB").resize((640, 640), Image.LANCZOS).save(PUB + f"/style_{i}_content.jpg", quality=84)
    res = Image.open(ST + r).convert("RGB")
    res.resize((640, 640), Image.LANCZOS).save(PUB + f"/style_{i}_result.jpg", quality=88)
    q = res.resize((120, 120)).quantize(colors=6, method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[:18]).reshape(6, 3)
    counts = np.bincount(np.array(q).ravel(), minlength=6)
    order = np.argsort(-counts)
    pairs.append(
        {"pos": pos, "palette": ["#%02x%02x%02x" % tuple(pal[j]) for j in order], "weights": [int(counts[j]) for j in order]}
    )
    print("style", i, pairs[-1]["palette"])
json.dump({"pairs": pairs}, open(DATA + "/style.json", "w"), indent=1)

# old single-image outputs
for f in ("lseg_input.jpg", "lseg_mask.png", "style_content.jpg", "style_result.jpg"):
    try:
        os.remove(os.path.join(PUB, f))
    except FileNotFoundError:
        pass
