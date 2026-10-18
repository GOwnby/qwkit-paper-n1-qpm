"""Self-consistent field curves for the figures: |chi(2)| vs F for the three symmetric families,
and the triple well's mechanism (subband energies, ground-state centroid, dipoles) vs F. Writes curves_sc.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from qwkit import sp, core as q
HW = 1.23984 / 10
# lattice-matched In0.53(Ga0.65Al0.35)0.47As step layers (linear interpolation, see stepwell.py)
q.MAT["q35"] = dict(V=0.52 * 0.35, m=0.043 + 0.032 * 0.35, Eg=0.75 + 0.70 * 0.35); sp.EPS_R["q35"] = 13.9 - 1.5 * 0.35
fam = {"Symmetric triple well 58/14/34/14/58": [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")],
       "Symmetric double well 56/6/56": [(56, "well"), (6, "barrier"), (56, "well")],
       "Single well 116": [(116, "well")],
       "Stepped well 50/40/50": [(50, "q35"), (40, "well"), (50, "q35")]}
AD = [(64, "well"), (10, "barrier"), (29, "well")]
ref = abs(sp.chi2_sc(AD, 0.0, HW, dz=0.5)["chi"])
out = dict(ref=ref, curves={}, mech=[])
for name, lay in fam.items():
    rows, VH = [], None
    Fmax = 110 if "triple" in name else 200
    for F in range(0, Fmax + 1, 5 if "triple" in name else 10):
        try:
            o = sp.chi2_sc(lay, float(F), HW, dz=0.5, VH0=VH); VH = o["r"]["VH"]
            # stop where the third subband is no longer bound (fewer than 3 localized states)
            if len(o["E"]) < 3:
                if "Stepped" in name: continue      # marginally bound third level near 110 kV/cm
                break
            rows.append(dict(F=F, pct=abs(o["chi"]) / ref * 100))
            if "triple" in name:
                Z = o["Z"]; zc = 0.0
                out["mech"].append(dict(F=F, E=[float(x) for x in (o["E"] - o["E"][0])[:4]], z11=float(Z[0, 0]),
                                        z13=float(abs(Z[0, 2])), z14=float(abs(Z[0, 3])), z34=float(abs(Z[2, 3])),
                                        z12=float(abs(Z[0, 1]))))
        except Exception as ex:
            break
    out["curves"][name] = rows
    print(name, "max %.1f%% at %s" % max((r["pct"], r["F"]) for r in rows), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves_sc.json"), "w"))
