"""N1, equal-freedom comparison at every doping and pump power (resonant pump, LO-phonon relaxation):
  TW   : sign reversal +-F, F in 30-90 kV/cm (valid fields only), best F per point (electrode factor 0.92);
  ADbip: ordinary ADQW with bipolar segments +-F, F in 20-120 kV/cm (valid only), best F per point (same electrode factor);
  Stark: ADQW 0 V / best valid off field (from stark_valid.json, electrode factor 1 as in Table 3).
Also the small-signal complex chi2 of every field state at the pump (phase difference of the bipolar segments).
Validity: no subband below E1 + 0.40 eV with active-region weight in (0.15, 0.8). Writes bipolar_fair.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, propagation as pr
import propagation_study as ps
import twofield_qpm as tq
here = os.path.dirname(os.path.abspath(__file__))
FT = list(range(30, 100, 10)); FA = list(range(20, 140, 20))


def chi(st):
    return complex(sp.chi2_depol(np.array(st["Et"]), np.array(st["Z"]), np.array(st["rho"]), ps.HW, 0.010, st["N"]))


def run(a):
    kind, i, sp_, sm, P = a
    m1 = pr.Medium(sp_["st"], ps.HW, R=sp_["R"]); m2 = pr.Medium(sm["st"], ps.HW, R=sm["R"])
    w1 = ps.wg_for(sp_["st"]); w2 = ps.wg_for(sm["st"])
    if kind == "tw":
        r = pr.propagate(m1, w1, P, "sign", eta_e=ps.ETA_E, L_max_um=5000.0)
    else:
        r = pr.propagate_segments([(m1, w1["Lcoh_um"], ps.ETA_E), (m2, w2["Lcoh_um"], ps.ETA_E)], ps.WG, P, L_max_um=5000.0)
    return dict(kind=kind, i=i, ns=ps.T["states"][i]["ns"], F=sp_["F"], P=P, eta=float(r["eta"]))


if __name__ == "__main__":
    n = len(ps.T["states"])
    with Pool(2) as pool:
        sts = pool.map(tq.state, [("tw", i, F) for i in range(n) for F in FT] + [("ad", i, s * F) for i in range(n) for F in FA for s in (1, -1)])
    S = {(s["design"], s["I"], s["F"]): s for s in sts}
    jobs = []; phases = []
    for i in range(n):
        for F in FT:
            s = S[("tw", i, F)]
            if s["valid"]:
                jobs += [("tw", i, s, s, P) for P in ps.PS]
        for F in FA:
            a, b = S[("ad", i, F)], S[("ad", i, -F)]
            ca, cb = chi(a["st"]), chi(b["st"])
            phases.append(dict(i=i, F=F, valid=bool(a["valid"] and b["valid"]), chi_plus=[abs(ca), float(np.degrees(np.angle(ca)))],
                               chi_minus=[abs(cb), float(np.degrees(np.angle(cb)))]))
            if a["valid"] and b["valid"]:
                jobs += [("ad", i, a, b, P) for P in ps.PS]
    print(len(jobs), "propagations", flush=True)
    with Pool(2) as pool:
        res = pool.map(run, jobs, chunksize=6)
    SV = json.load(open(os.path.join(here, "stark_valid.json")))["results"]
    best = []
    for i in range(n):
        for P in ps.PS:
            tw = max([r for r in res if r["kind"] == "tw" and r["i"] == i and r["P"] == P], key=lambda r: r["eta"])
            ad = max([r for r in res if r["kind"] == "ad" and r["i"] == i and r["P"] == P], key=lambda r: r["eta"])
            stk = [r for r in SV if abs(r["ns"] - ps.T["states"][i]["ns"]) < 1 and abs(r["P"] - P) < 1e-9][0]
            best.append(dict(ns=ps.T["states"][i]["ns"], P=P, tw=tw["eta"], F_tw=tw["F"], adbip=ad["eta"], F_ad=ad["F"], stark=stk["eta"], F_off=stk["F_off"]))
    json.dump(dict(results=res, best=best, phases=phases), open(os.path.join(here, "bipolar_fair.json"), "w"), indent=1)
    for b in best:
        if b["P"] in (ps.PS[0], ps.PS[7], ps.PS[10], ps.PS[11]):
            print(f"ns {b['ns']:.0e} P {b['P']:.2f}: TW {b['tw']*100:.4f}% (+-{b['F_tw']}) | ADQW bipolar {b['adbip']*100:.4f}% (+-{b['F_ad']}) | Stark {b['stark']*100:.4f}% ({b['F_off']:+d})"
                  f" | TW/Stark {b['tw']/b['stark']:.2f} TW/ADbip {b['tw']/b['adbip']:.2f}", flush=True)
    for p in phases:
        if p["i"] in (2, 4):
            print(p, flush=True)
