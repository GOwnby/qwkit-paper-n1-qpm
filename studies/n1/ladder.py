"""Model ladder and doping placement for the chosen designs: how much does each physical
ingredient move the Gate 1 ratio? Usage: python3 ladder.py tw0 tw1 tw2 F ad0 ad1 ad2"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from qwkit import sp
HW = 1.23984 / 10
a = [int(x) for x in sys.argv[1:8]]
TWg, F, ADg = a[0:3], float(a[3]), a[4:7]
def tw(g): return [(g[0], "well"), (g[1], "barrier"), (g[2], "well"), (g[1], "barrier"), (g[0], "well")]
def ad(g): return [(g[0], "well"), (g[1], "barrier"), (g[2], "well")]
def best_tw(**kw):
    best, bF, VH = 0, None, None
    for f in np.arange(F - 20, F + 25, 5):
        o = sp.chi2_sc(tw(TWg), float(f), HW, dz=0.5, VH0=VH, **kw)
        if kw.get("sc", True): VH = o["r"]["VH"]
        if abs(o["chi"]) > best: best, bF = abs(o["chi"]), float(f)
    return best, bF
rows = []
for label, kw in [("No Hartree, no depolarisation", dict(sc=False, depol=False)),
                  ("No Hartree, with depolarisation", dict(sc=False, depol=True)),
                  ("Hartree, no depolarisation", dict(sc=True, depol=False)),
                  ("Hartree + depolarisation (full)", dict(sc=True, depol=True)),
                  ("Full, donors in the wells", dict(sc=True, depol=True, doping="wells")),
                  ("Full, donors in the centre well", dict(sc=True, depol=True, doping="centre"))]:
    t, f = best_tw(**kw)
    kwa = dict(kw); 
    if kwa.get("doping") == "centre": kwa["doping"] = "wells"   # ADQW has no symmetric centre layer
    d = abs(sp.chi2_sc(ad(ADg), 0.0, HW, dz=0.5, **kwa)["chi"])
    rows.append(dict(model=label, tw=t, F=f, adqw=d, ratio=t / d))
    print(f"{label:34s} TW {t/1e3:6.1f} @ {f:.0f} | ADQW {d/1e3:6.1f} | ratio {t/d:.3f}", flush=True)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ladder.json"), "w"), indent=1)
