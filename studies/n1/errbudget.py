"""Gate 2 error budget for N1 with the self-consistent model.
For each parameter perturbation: re-tune the triple well's field (it is a free knob in the device),
keep the ADQW at 0 V, and report the ratio. Usage: python3 errbudget.py <tw g0 g1 g2> <ad g0 g1 g2>"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, core as q
HW = 1.23984 / 10; DZ = 1.0
TWg = tuple(int(x) for x in sys.argv[1:4]); ADg = tuple(int(x) for x in sys.argv[4:7])
def tw(g): return [(g[0], "well"), (g[1], "barrier"), (g[2], "well"), (g[1], "barrier"), (g[0], "well")]
def ad(g): return [(g[0], "well"), (g[1], "barrier"), (g[2], "well")]
BASE = {k: dict(v) for k, v in q.MAT.items()}
CASES = [("nominal", {}, 0.010, TWg, ADg),
         ("band offset -0.02 eV", {"barrier": {"V": 0.50}}, 0.010, TWg, ADg),
         ("band offset +0.02 eV", {"barrier": {"V": 0.54}}, 0.010, TWg, ADg),
         ("well mass -5%", {"well": {"m": 0.043 * 0.95}}, 0.010, TWg, ADg),
         ("well mass +5%", {"well": {"m": 0.043 * 1.05}}, 0.010, TWg, ADg),
         ("barrier mass -5%", {"barrier": {"m": 0.075 * 0.95}}, 0.010, TWg, ADg),
         ("barrier mass +5%", {"barrier": {"m": 0.075 * 1.05}}, 0.010, TWg, ADg),
         ("linewidth 6 meV", {}, 0.006, TWg, ADg),
         ("linewidth 14 meV", {}, 0.014, TWg, ADg)]
for i, lab in enumerate(["outer well", "barriers", "centre well"]):
    for d in (-3, 3):
        g = list(TWg); g[i] += d
        CASES.append((f"triple well {lab} {d:+d} A", {}, 0.010, tuple(g), ADg))
for i, lab in enumerate(["wide well", "barrier", "narrow well"]):
    for d in (-3, 3):
        g = list(ADg); g[i] += d
        CASES.append((f"ADQW {lab} {d:+d} A", {}, 0.010, TWg, tuple(g)))

def run(case):
    name, mods, G, tg, ag = case
    for k in q.MAT: q.MAT[k].clear(); q.MAT[k].update(BASE[k])
    for k, v in mods.items(): q.MAT[k].update(v)
    a = abs(sp.chi2_sc(ad(ag), 0.0, HW, gamma=G, dz=DZ)["chi"])
    best, bF, VH = 0.0, None, None
    for F in range(30, 95, 5):
        o = sp.chi2_sc(tw(tg), float(F), HW, gamma=G, dz=DZ, VH0=VH); VH = o["r"]["VH"]
        if abs(o["chi"]) > best: best, bF = abs(o["chi"]), F
    return dict(case=name, adqw=a, tw=best, F=bF)

if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(run, CASES)
    nom = res[0]
    for r in res:
        r["ratio_same_case"] = r["tw"] / r["adqw"]; r["tw_vs_nominal_adqw"] = r["tw"] / nom["adqw"]
        print(f"{r['case']:32s} TW {r['tw']/1e3:6.1f} nm/V @ {r['F']} kV/cm | ADQW {r['adqw']/1e3:6.1f} | ratio {r['ratio_same_case']:.2f}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "errbudget.json"), "w"), indent=1)
