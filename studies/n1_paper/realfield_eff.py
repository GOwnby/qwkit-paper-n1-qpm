"""N1 Table 3 and Fig. 5 with the electrode fields (realfield_common.py): resonant pump, LO-phonon relaxation, every
doping (1e10-5e11 cm^-2) and pump power (10 mW-20 W) of propagation_lo.json. Electrical schemes:
  TW 55    : triple-well sign reversal at +-55 kV/cm;
  TW opt   : triple-well sign reversal +-F, F = 10-90 kV/cm (5 kV/cm steps; 2F <= 188 kV/cm), best F;
  ADQW bip : ADQW bipolar segments +-F, F = 10-90 kV/cm (5 kV/cm steps; 2F <= 188 kV/cm breakdown limit), best F;
  Stark    : ADQW on at 0 V, off at the best valid field (|F| = 10-140 kV/cm, 5 kV/cm steps).
Each with realistic (fringing) fields; TW 55 and the best TW/ADQW patterns also with abrupt fields (no factor).
The growth-defined, idealized and unpatterned benchmarks do not involve electrodes and stay as in propagation_lo.json.
Also records SH power along the guide at 10 W, 5e11 cm^-2 for Fig. 5c. Writes realfield_eff.json."""
import os, sys, json, time
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import realfield_common as rc
import propagation_study as ps

PS = [float(p) for p in ps.PS]


def job(a):
    I, k = a; P = PS[k]; t0 = time.time()
    tw, ad = rc.bank("tw", I), rc.bank("ad", I)
    rec = (I == 4 and k == 10)
    out = dict(I=I, ns=rc.NS[I], P=P)
    out["tw55"] = dict(real=rc.ev(tw, [55, -55], P, I, record=rec), ideal=rc.ev(tw, [55, -55], P, I, real=False))
    cand = [F for F in range(10, 95, 5) if rc.has(tw, F) and rc.has(tw, -F) and rc.allowed(F, -F)]
    r = {F: rc.ev(tw, [F, -F], P, I)["eta"] for F in cand}
    Fb = max(r, key=r.get)
    out["twopt"] = dict(F=Fb, eta=r[Fb], all=r, real=rc.ev(tw, [Fb, -Fb], P, I), ideal_bestF=max(cand, key=lambda F: rc.ev(tw, [F, -F], P, I, real=False)["eta"]))
    cand = [F for F in range(10, 95, 5) if rc.has(ad, F) and rc.has(ad, -F) and rc.allowed(F, -F)]
    r = {F: rc.ev(ad, [F, -F], P, I)["eta"] for F in cand}
    Fb = max(r, key=r.get)
    out["adbip"] = dict(F=Fb, eta=r[Fb], all=r, real=rc.ev(ad, [Fb, -Fb], P, I, record=rec), ideal=rc.ev(ad, [Fb, -Fb], P, I, real=False))
    cand = [F for F in range(-140, 145, 5) if abs(F) >= 10 and rc.has(ad, F) and rc.allowed(0, F)]
    r = {F: rc.ev(ad, [0, F], P, I)["eta"] for F in cand}
    Fb = max(r, key=r.get)
    out["stark"] = dict(F=Fb, eta=r[Fb], all=r, real=rc.ev(ad, [0, Fb], P, I, record=rec), ideal=rc.ev(ad, [0, Fb], P, I, real=False))
    print(f"ns {rc.NS[I]:.0e} P {P:6.3f}: TW55 {100*out['tw55']['real']['eta']:.4f}% (abrupt {100*out['tw55']['ideal']['eta']:.4f}%) | "
          f"TW opt {100*out['twopt']['eta']:.4f}% +-{out['twopt']['F']} | ADQW bip {100*out['adbip']['eta']:.4f}% +-{out['adbip']['F']} | "
          f"Stark {100*out['stark']['eta']:.4f}% ({out['stark']['F']:+d}) | {time.time()-t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    jobs = [(I, k) for I in (4, 2, 3, 1, 0) for k in range(len(PS))]
    res = rc.run_jobs(job, jobs, "eff")
    json.dump(res, open(os.path.join(here, "realfield_eff.json"), "w"))
