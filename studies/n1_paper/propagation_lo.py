"""Saturated, pump-depleted SHG with level-resolved LO-phonon relaxation (rates.json) instead of a single
1 ps relaxation time. Same schemes, powers and doping levels as propagation_study.py + adqw_sign.py.
Writes propagation_lo.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
here = os.path.dirname(os.path.abspath(__file__))
RATES = json.load(open(os.path.join(here, "rates.json")))
def R_for(i, d):
    return [r for r in RATES if r["i"] == i and r["design"] == d][0]["R"]

def job(args):
    i, which, scheme, P = args
    stset = ps.T["states"][i]
    st = stset[which]
    med = pr.Medium(st, ps.HW, gamma=0.010, R=R_for(i, which))
    w = ps.wg_for(st)
    eta_e = ps.ETA_E if (which == "tw" and scheme == "sign") else 1.0
    r = pr.propagate(med, w, P, scheme, eta_e=eta_e, L_max_um=5000.0)
    return dict(ns=stset["ns"], design=which, scheme=scheme, P=P, eta=float(r["eta"]), L_opt_um=float(r["L_opt_um"]),
                Lcoh_um=w["Lcoh_um"])

if __name__ == "__main__":
    jobs = []
    for i in range(len(ps.T["states"])):
        for P in ps.PS:
            jobs += [(i, "ad", "none", P), (i, "ad", "onoff", P), (i, "ad", "perfect", P), (i, "ad", "sign", P), (i, "tw", "sign", P)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=3)
    json.dump(dict(results=res, wg=ps.WG, eta_e=ps.ETA_E, model="LO-phonon rates"), open(os.path.join(here, "propagation_lo.json"), "w"))
    print("done", len(res))
