"""Sensitivity of the propagation comparison to the linewidth (6, 10, 14 meV) and to the relaxation model
(LO-phonon rates; LO rates plus an extra 1/(1 ps) channel to the ground state mimicking interface-roughness
and electron-electron scattering; single 1 ps time). Ratios of the triple well with sign reversal to the
ADQW with growth-defined orientation reversal and with Stark on/off QPM (off field from stark_valid.json: the best valid
field at the baseline linewidth and relaxation model, held fixed for the variations).
Writes sensitivity.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
import twofield_qpm as tq
here = os.path.dirname(os.path.abspath(__file__))
RATES = json.load(open(os.path.join(here, "rates.json")))
OFF = {}

def R_model(R, model):
    R = np.array(R)
    if model == "LO":
        return R
    if model == "LO+1ps":
        R = R.copy(); R[1:, 0] += 1e12
        return R
    return None

def job(a):
    i, P, g, model = a
    on = ps.T["states"][i]["ad"]; tw = ps.T["states"][i]["tw"]
    Rget = lambda d: [r for r in RATES if r["i"] == i and r["design"] == d][0]["R"]
    off = OFF[(i, P)]
    mk = lambda st, R: pr.Medium(st, ps.HW, gamma=g, R=R_model(R, model))
    m_tw, m_on, m_off = mk(tw, Rget("tw")), mk(on, Rget("ad")), mk(off["st"], off["R"])
    w_tw, w_on, w_off = ps.wg_for(tw), ps.wg_for(on), ps.wg_for(off["st"])
    e_tw = pr.propagate(m_tw, w_tw, P, "sign", eta_e=ps.ETA_E)["eta"]
    e_rev = pr.propagate(m_on, w_on, P, "sign", eta_e=1.0)["eta"]
    e_st = pr.propagate_segments([(m_on, w_on["Lcoh_um"], 1.0), (m_off, w_off["Lcoh_um"], 1.0)], ps.WG, P)["eta"]
    return dict(ns=ps.T["states"][i]["ns"], P=P, gamma=g, model=model, tw=float(e_tw), rev=float(e_rev), stark=float(e_st))

if __name__ == "__main__":
    SV = json.load(open(os.path.join(here, "stark_valid.json")))["results"]
    for i in (2, 4):
        for P in (0.01, 1.26, 10.0):
            F = min([r for r in SV if abs(r["ns"] - ps.T["states"][i]["ns"]) < 1], key=lambda r: abs(r["P"] - P))["F_off"]
            OFF[(i, P)] = tq.state(("ad", i, F))
    jobs = [(i, P, g, m) for i in (2, 4) for P in (0.01, 1.26, 10.0) for g in (0.006, 0.010, 0.014) for m in ("LO", "LO+1ps", "tau1")
            if not (g != 0.010 and m != "LO")]
    with Pool(2) as p:
        res = p.map(job, jobs)
    json.dump(res, open(os.path.join(here, "sensitivity.json"), "w"))
    for r in res:
        print(f"ns {r['ns']:.0e} P {r['P']:5.2f} G {r['gamma']*1e3:.0f} meV {r['model']:7s}: TW {r['tw']*100:.4f}%  TW/rev {r['tw']/r['rev']:.2f}  TW/Stark {r['tw']/r['stark']:.2f}")
