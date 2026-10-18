"""Digitize the 18-vertex graph figure of the quantum-communication project (src/assets/quantum_communication.png)
into vertices + edges for the tile. Vertices = thick black blobs (lines are thin, so a binary opening removes them);
edges = vertex pairs whose connecting segment is (almost) entirely dark in the image.

Run: python scripts/tile-data/quantum_graph.py   (needs numpy, scipy, pillow)
The numeric labels are not read from the image; vertices are matched to the figure's 1..18 by hand (LABELS below, checked visually).
"""
import itertools, json, os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(HERE, "../../src/assets/quantum_communication.png")).convert("RGBA")
bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
bg.alpha_composite(img)
dark = np.array(bg.convert("L")) < 110

# vertices: opening with a disc removes strokes (~3px) but keeps dots (~14px)
r = 5
yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
disc = (xx * xx + yy * yy) <= r * r
blobs = ndi.binary_opening(dark, structure=disc)
lab, n = ndi.label(blobs)
cent = np.array(ndi.center_of_mass(blobs, lab, range(1, n + 1)))[:, ::-1]  # (x, y)
print("vertices found:", n)
assert n == 18, "expected 18 vertices"

# order by the figure's numbering — vertex numbers read off the figure, approximate pixel positions
FIG = {1: (90, 363), 2: (252, 224), 3: (206, 202), 4: (367, 87), 5: (481, 224), 6: (529, 202), 7: (643, 363),
       8: (505, 477), 9: (529, 523), 10: (366, 638), 11: (228, 477), 12: (207, 523), 13: (297, 363), 14: (436, 363),
       15: (781, 157), 16: (895, 157), 17: (781, 523), 18: (895, 523)}
order = {}
for k, (x, y) in FIG.items():
    d = np.hypot(cent[:, 0] - x, cent[:, 1] - y)
    order[k] = int(d.argmin())
    assert d.min() < 25, (k, d.min())
assert len(set(order.values())) == 18

# edges: fraction of dark pixels along the segment (skip the dot radius at both ends)
def frac(a, b):
    p, q = cent[a], cent[b]
    L = np.hypot(*(q - p))
    t = np.linspace(14 / L, 1 - 14 / L, int(L))
    xs = (p[0] + (q[0] - p[0]) * t).round().astype(int)
    ys = (p[1] + (q[1] - p[1]) * t).round().astype(int)
    # tolerate 1px misalignment
    d = ndi.binary_dilation(dark, iterations=1)
    return d[ys, xs].mean()

edges = []
for i, j in itertools.combinations(range(1, 19), 2):
    f = frac(order[i], order[j])
    if f > 0.985:
        edges.append((i, j))
# drop an edge a–c that is just two collinear edges a–b, b–c (vertex b sits on the segment)
def on_segment(a, c, b, tol=6):
    p, q, r = cent[order[a]], cent[order[c]], cent[order[b]]
    d = q - p
    t = np.dot(r - p, d) / np.dot(d, d)
    return 0.05 < t < 0.95 and np.hypot(*(p + t * d - r)) < tol

E = set(map(tuple, edges))
edges = [(a, c) for a, c in edges
         if not any(on_segment(a, c, b) and tuple(sorted((a, b))) in E and tuple(sorted((b, c))) in E for b in range(1, 19) if b not in (a, c))]
print(len(edges), "edges")

# 4-cliques ("contexts")
adj = {i: set() for i in range(1, 19)}
for i, j in edges:
    adj[i].add(j); adj[j].add(i)
cliques = [c for c in itertools.combinations(range(1, 19), 4) if all(b in adj[a] for a, b in itertools.combinations(c, 2))]
print("4-cliques:", cliques)

W, H = img.size
verts = {k: [round(float(cent[order[k]][0]) / W, 4), round(float(cent[order[k]][1]) / H, 4)] for k in FIG}
out = os.path.join(HERE, "../../src/data/tiles/quantum.json")
json.dump({"aspect": W / H, "v": verts, "e": edges, "k4": cliques}, open(out, "w"), separators=(",", ":"))
print("->", os.path.normpath(out))
