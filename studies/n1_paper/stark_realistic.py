"""Realistic Stark on/off QPM for the ADQW: 'on' segments at 0 V and 'off' segments at the smallest |F|
(either polarity) where |chi(2)| <= 10% of the 0 V value, each with its own self-consistent chi(2),
absorption, ISB refractive index (hence coherence length) and LO-phonon relaxation. Also a check that the
sign-reversal propagation with propagate_segments reproduces propagate(). Writes stark_realistic.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, scattering as sc, propagation as pr
import propagation_study as ps
here = os.path.dirname(os.path.abspath(__file__))
AD = [(64, "well"), (10, "barrier"), (29, "well")]
RATES = json.load(open(os.path.join(here, "rates.json")))

def state(F, ns):
    o = sp.chi2_sc(AD, float(F), ps.HW, ns_cm2=ns)
    E, psi, Z, rho = sp.prepare(o["r"])
    st = dict(E=o["E"].tolist(), Z=o["Z"].tolist(), Et=o["Et"].tolist(), rho=o["rho"].tolist(), N=o["N"])
    return st, sc.rate_matrix(o["r"]["z"], psi, E), abs(o["chi"])

def off_state(i):
    ns = ps.T["states"][i]["ns"]
    c_on = np.hypot(*ps.T["states"][i]["ad"]["chi"])
    for Fa in range(10, 260, 10):
        for F in (Fa, -Fa):
            st, R, c = state(F, ns)
            if c <= 0.1 * c_on:
                return dict(i=i, ns=ns, F_off=F, chi_ratio=c / c_on, st=st, R=R.tolist())
    raise RuntimeError("no off state")

def job(a):
    i, P, off = a
    on_st = ps.T["states"][i]["ad"]; Ron = [r for r in RATES if r["i"] == i and r["design"] == "ad"][0]["R"]
    m_on = pr.Medium(on_st, ps.HW, R=Ron); m_off = pr.Medium(off["st"], ps.HW, R=off["R"])
    w_on = ps.wg_for(on_st); w_off = ps.wg_for(off["st"])
    r = pr.propagate_segments([(m_on, w_on["Lcoh_um"], 1.0), (m_off, w_off["Lcoh_um"], 1.0)], ps.WG, P, L_max_um=5000.0)
    return dict(ns=off["ns"], P=P, eta=float(r["eta"]), L_opt_um=float(r["L_opt_um"]), Lcoh_on=w_on["Lcoh_um"], Lcoh_off=w_off["Lcoh_um"])

if __name__ == "__main__":
    # consistency check: sign reversal via propagate_segments == propagate
    i = 4; st = ps.T["states"][i]["tw"]; Rt = [r for r in RATES if r["i"] == i and r["design"] == "tw"][0]["R"]
    m = pr.Medium(st, ps.HW, R=Rt); w = ps.wg_for(st)
    a = pr.propagate(m, w, 1.26, "sign", eta_e=1.0, L_max_um=500.0)["eta"]
    b = pr.propagate_segments([(m, w["Lcoh_um"], 1.0), (m, w["Lcoh_um"], -1.0)], w, 1.26, L_max_um=500.0)["eta"]
    print(f"check propagate vs propagate_segments: {a:.6e} {b:.6e}")
    with Pool(2) as p:
        offs = p.map(off_state, range(len(ps.T["states"])))
    for o in offs:
        print(f"ns {o['ns']:.0e}: off field {o['F_off']} kV/cm, |chi| off/on {o['chi_ratio']:.3f}")
    jobs = [(o["i"], P, o) for o in offs for P in ps.PS]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=3)
    json.dump(dict(results=res, offs=[dict(ns=o["ns"], F_off=o["F_off"], chi_ratio=o["chi_ratio"]) for o in offs]),
              open(os.path.join(here, "stark_realistic.json"), "w"))
    for r in res:
        if r["P"] in (ps.PS[0], ps.PS[7], ps.PS[10]):
            print(f"ns {r['ns']:.0e} P {r['P']:.2f}: realistic Stark eta {r['eta']*100:.4f}% L {r['L_opt_um']:.0f} um (Lcoh on/off {r['Lcoh_on']:.1f}/{r['Lcoh_off']:.1f})")
