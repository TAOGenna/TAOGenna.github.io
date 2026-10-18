"""Run the repo's real ionogram inversion and dump every step as JSON for the Inversion tile.

Re-uses make_animation.main() from the inversion repo, patching save_frame so each step's
state (label, L2 error, reconstructed profile, reconstructed ionogram) is recorded instead of
rendered. Output: src/data/tiles/inversion.json.
Run: python scripts/tile-data/inversion.py   (needs numpy, scipy, matplotlib, tqdm)
"""
import contextlib, io, json, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))  # captured before chdir
REPO = os.path.expanduser("~/Desktop/kenyi/projects/inversion-algorithm-plasma-frequency-profile")
os.chdir(REPO)  # the repo uses relative paths
sys.path.insert(0, REPO)
import make_animation as ma  # noqa: E402
from src.get_ionogram import get_ionogram  # noqa: E402

rec = {"measured": None, "steps": [], "lim": None}


def thin(a, n):  # keep n evenly spaced samples
    a = np.asarray(a, dtype=float)
    return a if len(a) <= n else a[np.linspace(0, len(a) - 1, n).astype(int)]


def save_frame(idx, label, error, frq, vh, fp, rh, step, total_steps, x_profile_lim=None, y_profile_lim=None):
    if rec["measured"] is None:
        ok = np.isfinite(frq) & np.isfinite(vh)  # the SAO trace has NaN gaps
        rec["measured"] = {"f": [round(float(x), 3) for x in frq[ok]], "h": [round(float(x), 1) for x in vh[ok]]}
        rec["lim"] = {"fp": [float(v) for v in x_profile_lim], "rh": [float(v) for v in y_profile_lim]}
    if fp is None:  # the "input ionogram" frame
        return
    s_f, s_vh = get_ionogram(rh, fp, fp)
    ok = np.isfinite(s_f) & np.isfinite(s_vh)
    s_f, s_vh = np.asarray(s_f)[ok], np.asarray(s_vh)[ok]
    rec["steps"].append({
        "label": label, "err": round(float(error), 4),
        "fp": [round(float(x), 3) for x in thin(fp, 60)], "rh": [round(float(x), 1) for x in thin(rh, 60)],
        "rf": [round(float(x), 3) for x in thin(s_f, 70)], "rv": [round(float(x), 1) for x in thin(s_vh, 70)],
    })


ma.save_frame = save_frame
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ma.main()
rec["sounding"] = re.search(r"Using ionogram: (.+?)  \(", buf.getvalue()).group(1)  # e.g. 'Sat May 11 12:30:00 2024'

out = os.path.join(HERE, "../../src/data/tiles/inversion.json")
json.dump(rec, open(out, "w"), separators=(",", ":"))
assert len(rec["steps"]) > 3 and all(np.isfinite(s[k]).all() for s in rec["steps"] for k in ("fp", "rh", "rf", "rv"))
assert np.isfinite(rec["measured"]["h"]).all()
print(len(rec["steps"]), "steps,", len(rec["measured"]["f"]), "measured points ->", out, round(os.path.getsize(out) / 1024, 1), "KB")
