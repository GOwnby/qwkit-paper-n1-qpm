"""Extra benchmark: ADQW with ideal orientation-reversed QPM (growth-defined sign reversal, eta_e = 1,
no regrowth loss). Appends results to propagation.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
here = os.path.dirname(os.path.abspath(__file__))
def job(a):
    i, P = a
    st = ps.T["states"][i]
    med = pr.Medium(st["ad"], ps.HW); w = ps.wg_for(st["ad"])
    r = pr.propagate(med, w, P, "sign", eta_e=1.0, L_max_um=5000.0)
    return dict(ns=st["ns"], design="ad", scheme="sign", P=P, eta=float(r["eta"]), L_opt_um=float(r["L_opt_um"]), Lcoh_um=w["Lcoh_um"])
if __name__ == "__main__":
    D = json.load(open(os.path.join(here, "propagation.json")))
    D["results"] = [r for r in D["results"] if not (r["design"] == "ad" and r["scheme"] == "sign")]
    with Pool(2) as p:
        res = p.map(job, [(i, P) for i in range(len(ps.T["states"])) for P in ps.PS])
    D["results"] += res
    json.dump(D, open(os.path.join(here, "propagation.json"), "w"))
    R = D["results"]
    for ns in sorted(set(r["ns"] for r in R)):
        for P in (ps.PS[0], ps.PS[7], ps.PS[10]):
            g = lambda d, s: [r for r in R if r["ns"] == ns and r["P"] == P and r["design"] == d and r["scheme"] == s][0]["eta"]
            print(f"ns {ns:.0e} P {P:.2f}: TW sign {g('tw','sign')*100:.3f}% | ADQW orient-reversed {g('ad','sign')*100:.3f}% | ratio TW/ADQWrev {g('tw','sign')/g('ad','sign'):.2f}")
