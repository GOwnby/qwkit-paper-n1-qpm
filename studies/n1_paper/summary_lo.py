"""Collects the numbers of the first N1 release from propagation_lo.json, stark_valid.json, sensitivity.json
and rates.json (electrical schemes scaled by the 0.92 electrode factor); estimates heating. Writes summary_lo.json.
The electrically written schemes of the current manuscript come from realfield_summary.py (electrode fields and
breakdown limit); the benchmarks without electrodes quoted there are those of propagation_lo.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from qwkit import propagation as pr
import propagation_study as ps
here = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(here, "propagation_lo.json")))["results"]
S = json.load(open(os.path.join(here, "stark_valid.json")))
B += [dict(r, design="ad", scheme="stark") for r in S["results"]]
RATES = json.load(open(os.path.join(here, "rates.json")))
def g(ns, P, d, s, key="eta"):
    return [r for r in B if abs(r["ns"] - ns) < 1 and abs(r["P"] - P) < 1e-6 and r["design"] == d and r["scheme"] == s][0][key]
NS = sorted(set(r["ns"] for r in B)); PS = sorted(set(r["P"] for r in B))
out = {}
rows = []
for P in (PS[7], PS[10]):
    for ns in (1e10, 1e11, 5e11):
        rows.append(dict(P=P, ns=ns, tw=g(ns, P, "tw", "sign"), stark=g(ns, P, "ad", "stark"), onoff=g(ns, P, "ad", "onoff"),
                         rev=g(ns, P, "ad", "sign"), none=g(ns, P, "ad", "none"), pm=g(ns, P, "ad", "perfect"),
                         L=g(ns, P, "tw", "sign", "L_opt_um")))
out["table"] = rows
for r in rows:
    print(f"{r['P']:.2f} W ns {r['ns']:.0e}: TW {r['tw']*100:.3f} | Stark {r['stark']*100:.3f} | onoff(on-loss) {r['onoff']*100:.3f} | rev {r['rev']*100:.3f} | none {r['none']*100:.4f} | PM {r['pm']*100:.3f} | L {r['L']:.0f} um")
def rng(num, den, cond):
    v = [g(ns, P, *num) / g(ns, P, *den) for ns in NS for P in PS if cond(P)]
    return min(v), max(v)
for lab, den in (("stark", ("ad", "stark")), ("onoff", ("ad", "onoff")), ("rev", ("ad", "sign")), ("pm", ("ad", "perfect"))):
    hi = rng(("tw", "sign"), den, lambda P: P >= 1.2); lo = rng(("tw", "sign"), den, lambda P: P < 0.011)
    out[f"ratio_{lab}"] = dict(high=hi, low=lo)
    print(f"TW/{lab}: P>=1.26 W {hi[0]:.2f}-{hi[1]:.2f}; 10 mW {lo[0]:.2f}-{lo[1]:.2f}")
# crossover power vs realistic Stark and vs rev for each ns
for lab, den in (("stark", ("ad", "stark")), ("rev", ("ad", "sign"))):
    cr = {}
    for ns in NS:
        c = next((P for P in PS if g(ns, P, "tw", "sign") > g(ns, P, *den)), None)
        cr[f"{ns:.0e}"] = c
    out[f"crossover_{lab}"] = cr; print("crossover", lab, {k: (None if v is None else round(v, 3)) for k, v in cr.items()})
# saturation intensities (absorption halving at w) and lifetimes at 5e11
for r in json.load(open(os.path.join(here, "rates.json"))):
    pass
# heating: absorbed pump power over the optimum length at 10 W, 5e11, LO model
i = [k for k, s in enumerate(ps.T["states"]) if abs(s["ns"] - 5e11) < 1][0]
st = ps.T["states"][i]["tw"]; R = [r for r in RATES if r["i"] == i and r["design"] == "tw"][0]["R"]
m = pr.Medium(st, ps.HW, R=R); w = ps.wg_for(st)
res = pr.propagate(m, w, 10.0, "sign", eta_e=ps.ETA_E, L_max_um=5000.0)
k = int(np.argmax(res["P2"])); L = res["x_um"][k] * 1e-4                 # cm
Pabs = 10.0 - res["P1"][k] - res["P2"][k]
V = 40 * 378e-8 * 15e-4 * L                                                # cm^3 (stack volume)
q = Pabs / V; Cv = 1.5                                                     # W/cm^3, J/(cm^3 K)
out["heating"] = dict(L_um=L * 1e4, Pabs_W=Pabs, q_Wcm3=q, dTdt_K_per_ns=q / Cv * 1e-9)
print(f"heating at 10 W, 5e11: L {L*1e4:.0f} um, absorbed {Pabs:.2f} W, {q:.2e} W/cm3, adiabatic {q/Cv*1e-9:.2f} K/ns")
json.dump(out, open(os.path.join(here, "summary_lo.json"), "w"), indent=1)
