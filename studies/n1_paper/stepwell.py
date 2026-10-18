"""Benchmark the symmetric stepped well (barrier | InGaAlAs step | InGaAs core | step | barrier), the
symmetric structure class of U.S. Patent 5,289,309, under the same conditions as the triple well.
Quaternary In0.53(Ga1-xAlx)0.47As parameters are interpolated linearly between the lattice-matched
ternaries (V = 0.52x eV, m = 0.043 + 0.032x, Eg = 0.75 + 0.70x, eps = 13.9 - 1.5x).
Coarse search with fixed potential and no depolarisation, then the full self-consistent model for the
best candidates. Writes stepwell.json."""
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import core as q, sp
here = os.path.dirname(os.path.abspath(__file__))
HW = 1.23984 / 10
XS = (0.2, 0.35, 0.5)
for x in XS:
    k = f"q{int(round(100*x))}"
    q.MAT[k] = dict(V=0.52 * x, m=0.043 + 0.032 * x, Eg=0.75 + 0.70 * x)
    sp.EPS_R[k] = 13.9 - 1.5 * x

def lay(tc, ts, x):
    k = f"q{int(round(100*x))}"
    return [(ts, k), (tc, "well"), (ts, k)]

def coarse(a):
    tc, ts, x = a
    best = (0.0, None)
    for F in range(15, 165, 15):
        try:
            o = sp.chi2_sc(lay(tc, ts, x), float(F), HW, sc=False, depol=False)
        except Exception:
            continue
        if abs(o["chi"]) > best[0]:
            best = (abs(o["chi"]), F)
    return dict(tc=tc, ts=ts, x=x, chi=best[0], F=best[1])

def full(c):
    out = []
    for F in range(max(10, c["F"] - 30), c["F"] + 35, 5):
        o = sp.chi2_sc(lay(c["tc"], c["ts"], c["x"]), float(F), HW, ns_cm2=5e11)
        out.append((F, abs(o["chi"])))
    F, v = max(out, key=lambda t: t[1])
    return dict(c, F_full=F, chi_full=v, curve=out)

if __name__ == "__main__":
    grid = list(itertools.product(range(20, 90, 10), range(10, 70, 10), XS))
    with Pool(2) as p:
        C = p.map(coarse, grid)
    C.sort(key=lambda r: -r["chi"])
    print("coarse top 5:", [(r["tc"], r["ts"], r["x"], r["F"], round(r["chi"] / 1e3, 1)) for r in C[:5]])
    with Pool(2) as p:
        Fu = p.map(full, C[:6])
    Fu.sort(key=lambda r: -r["chi_full"])
    # optimized ADQW at 5e11 cm^-2 in the same full model (tables.json, written by tables.py; run that first)
    _t = [r for r in json.load(open(os.path.join(here, "tables.json")))["states"] if abs(r["ns"] - 5e11) < 1]
    AD = float(np.hypot(*_t[0]["ad"]["chi"]))
    for r in Fu:
        print(f"step well core {r['tc']} A, steps {r['ts']} A, x={r['x']}: |chi| {r['chi_full']/1e3:.1f} nm/V at {r['F_full']} kV/cm = {100*r['chi_full']/AD:.0f}% of ADQW")
    json.dump(dict(coarse=C[:20], full=Fu), open(os.path.join(here, "stepwell.json"), "w"))
