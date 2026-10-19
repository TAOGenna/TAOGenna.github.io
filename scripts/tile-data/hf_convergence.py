"""Self-consistency residual per iteration, from the Hartree-Fock repo's own SCF (make_animation.py).

Runs the same loop as make_animation.scf (N_chain=24, eps_d, V from that file) and records the
expectation values after every iteration -> residual |evs_k - evs_{k-1}| (max-norm). Writes
src/data/tiles/hf_convergence.json for the Writing mockups.
Run: python scripts/tile-data/hf_convergence.py   (needs numpy + matplotlib)
"""
import json, os, sys

import numpy as np

REPO = os.path.expanduser("~/Desktop/kenyi/projects/Hartree-Fock-hybrid-QDsuperconductor")
sys.path.insert(0, REPO)
import make_animation as ma  # noqa: E402

N = 24
H = ma.build_H(N, ma.EPS_D)
evs = np.array([0.133222, 0.315101, 0.531797, 0.489767])  # same starting guess as ma.scf
hist = [evs.copy()]
for _ in range(ma.SCF_ITER):
    ma.apply_HF(H, evs, ma.V)
    w, v = np.linalg.eigh(H)
    new = np.zeros(4)
    for i in range(len(w)):
        if w[i] < 0.0:
            f = 0.5 if abs(w[i]) < 1e-8 else 1.0
            new[0] += np.conj(v[0, i]) * v[3, i] * f
            new[1] += np.conj(v[0, i]) * v[2, i] * f
            new[2] += np.conj(v[0, i]) * v[0, i] * f
            new[3] += np.conj(v[2, i]) * v[2, i] * f
    ma.apply_HF(H, evs, -ma.V)
    evs = new.real
    hist.append(evs.copy())

hist = np.array(hist)
res = np.abs(np.diff(hist, axis=0)).max(axis=1)
out = {"N": N, "V": ma.V, "eps_d": ma.EPS_D, "residual": [max(float(r), 1e-16) for r in res], "n_d": [float(x) for x in hist[1:, 2]]}
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/data/tiles/hf_convergence.json")
json.dump(out, open(dst, "w"), separators=(",", ":"))
print(len(res), "iterations; residual", f"{res[0]:.2e}", "->", f"{res[-1]:.2e}", "| n_d final", round(hist[-1, 2], 4))
