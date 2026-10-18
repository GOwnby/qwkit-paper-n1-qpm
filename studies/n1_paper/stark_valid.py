"""N1 benchmark revision: Stark on/off QPM with the 'on' segments at 0 V and the 'off' field chosen, for every doping
and pump power, as the field (both polarities, |F| <= 140 kV/cm) that MAXIMIZES the conversion efficiency among the
fields where the single-period model is valid (no partially localized subband below E1 + 0.40 eV; see
stark_offscan.py). Replaces stark_realistic.py, whose off fields (lowest field with |chi2| < 10%, 120-140 kV/cm)
lay where hybridized upper subbands were discarded. Same output format as stark_realistic.json plus the chosen field.
Writes stark_valid.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
import twofield_qpm as tq
here = os.path.dirname(os.path.abspath(__file__))
RATES = json.load(open(os.path.join(here, "rates.json")))
FA = [s * f for f in range(20, 160, 20) for s in (1, -1)]


def job(a):
    i, P, off = a
    on_st = ps.T["states"][i]["ad"]; Ron = [r for r in RATES if r["i"] == i and r["design"] == "ad"][0]["R"]
    m_on = pr.Medium(on_st, ps.HW, R=Ron); m_off = pr.Medium(off["st"], ps.HW, R=off["R"])
    w_on = ps.wg_for(on_st); w_off = ps.wg_for(off["st"])
    r = pr.propagate_segments([(m_on, w_on["Lcoh_um"], 1.0), (m_off, w_off["Lcoh_um"], 1.0)], ps.WG, P, L_max_um=5000.0)
    return dict(ns=ps.T["states"][i]["ns"], P=P, F_off=off["F"], eta=float(r["eta"]), L_opt_um=float(r["L_opt_um"]),
                Lcoh_on=w_on["Lcoh_um"], Lcoh_off=w_off["Lcoh_um"])


if __name__ == "__main__":
    n = len(ps.T["states"])
    with Pool(2) as pool:
        sts = pool.map(tq.state, [("ad", i, F) for i in range(n) for F in FA])
    valid = {i: [s for s in sts if s["I"] == i and s["valid"]] for i in range(n)}
    for i in range(n):
        print(f"ns {ps.T['states'][i]['ns']:.0e}: valid off fields {[s['F'] for s in valid[i]]}", flush=True)
    jobs = [(i, P, o) for i in range(n) for P in ps.PS for o in valid[i]]
    with Pool(2) as pool:
        allres = pool.map(job, jobs, chunksize=6)
    best = []
    for i in range(n):
        for P in ps.PS:
            rr = [r for r in allres if r["ns"] == ps.T["states"][i]["ns"] and r["P"] == P]
            best.append(max(rr, key=lambda r: r["eta"]))
    json.dump(dict(results=best, all=allres, rule="best valid off field per doping and power"),
              open(os.path.join(here, "stark_valid.json"), "w"))
    for r in best:
        if r["P"] in (ps.PS[0], ps.PS[7], ps.PS[10], ps.PS[11]):
            print(f"ns {r['ns']:.0e} P {r['P']:.2f}: Stark (best valid off field {r['F_off']:+d}) eta {r['eta']*100:.4f}%", flush=True)
