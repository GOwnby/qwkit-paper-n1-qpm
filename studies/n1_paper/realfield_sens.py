"""N1 Table 6 (sensitivity to the linewidth and the relaxation model) with the electrode fields (realfield_common.py):
triple well at +-55 kV/cm and Stark on/off QPM with the off field held at its baseline optimum (realfield_eff.json),
for Gamma = 6, 10, 14 meV with LO-phonon rates, and LO rates plus a 1 ps channel to the ground state, and a single
1 ps time, at Gamma = 10 meV (as sensitivity.py). Coherence lengths are those of the baseline (as in sensitivity.py).
Growth reversal involves no electrodes; its values are taken from sensitivity.json. Writes realfield_sens.json."""
import os, sys, json, copy
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import realfield_common as rc
from qwkit import propagation as pr
import common_n2 as c

EFF = json.load(open(os.path.join(here, "realfield_eff.json")))
SENS = json.load(open(os.path.join(here, "sensitivity.json")))


def R_model(R, model):
    R = np.array(R)
    if model == "LO":
        return R
    if model == "LO+1ps":
        R = R.copy(); R[1:, 0] += 1e12
        return R
    return None


def vbank(design, I, g, model):
    b = copy.copy(rc.bank(design, I)); b._mix = {}
    sts = c.statesets(design, I); hw = c.HW0
    b.media = [pr.Medium(s["st"], hw, gamma=g, R=R_model(s["R"], model)) for s in sts]
    b.dk = np.pi / (b.Lcoh * 1e-6)
    return b


def job(a):
    I, P, g, model = a
    e = min([r for r in EFF if r["I"] == I], key=lambda r: abs(r["P"] - P))
    Foff = e["stark"]["F"]
    tw = rc.ev(vbank("tw", I, g, model), [55, -55], P, I)["eta"]
    st = rc.ev(vbank("ad", I, g, model), [0, Foff], P, I)["eta"]
    rev = [r for r in SENS if abs(r["ns"] - rc.NS[I]) < 1 and abs(r["P"] - P) < 1e-9 and abs(r["gamma"] - g) < 1e-9 and r["model"] == model][0]["rev"]
    print(f"ns {rc.NS[I]:.0e} P {P} G {1e3*g:.0f} {model}: TW {100*tw:.4f}% TW/rev {tw/rev:.2f} TW/Stark {tw/st:.2f} (off {Foff:+d})", flush=True)
    return dict(I=I, ns=rc.NS[I], P=P, gamma=g, model=model, tw=tw, stark=st, rev=rev, F_off=Foff)


if __name__ == "__main__":
    jobs = [(I, P, g, m) for I in (2, 4) for P in (0.01, 1.26, 10.0) for g in (0.006, 0.010, 0.014) for m in ("LO", "LO+1ps", "tau1")
            if not (g != 0.010 and m != "LO")]
    res = rc.run_jobs(job, jobs, "sens")
    json.dump(res, open(os.path.join(here, "realfield_sens.json"), "w"))
