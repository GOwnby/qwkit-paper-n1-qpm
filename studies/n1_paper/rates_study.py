"""Level-resolved LO-phonon relaxation for the state sets of tables.json (triple well at its optimum
field, ADQW at 0 V), effective saturation intensities of the pump transition with the single-tau and
LO-phonon models. Writes rates.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, scattering as sc, propagation as pr
here = os.path.dirname(os.path.abspath(__file__))
HW = 1.23984 / 10
TW = [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")]
AD = [(64, "well"), (10, "barrier"), (29, "well")]
T = json.load(open(os.path.join(here, "tables.json")))

def rates(args):
    i, which = args
    s = T["states"][i]
    lay, F = (TW, s["F_tw"]) if which == "tw" else (AD, 0)
    o = sp.chi2_sc(lay, float(F), HW, ns_cm2=s["ns"], dz=0.5)
    E, psi, Z, rho = sp.prepare(o["r"])
    ref = np.array(s[which]["E"])
    R = sc.rate_matrix(o["r"]["z"], psi, E)
    return dict(i=i, design=which, ns=s["ns"], R=R.tolist(), dE_max_meV=float(np.max(np.abs(E - ref)) * 1e3))

def isat(med, I_grid):
    """Pump intensity (W/m^2) at which the ISB absorption at w falls to half its small-signal value."""
    a0 = med.alpha(med.populations(0.0, 0.0))[0]
    a = np.array([med.alpha(med.populations(I, 0.0))[0] for I in I_grid])
    k = np.argmax(a < 0.5 * a0)
    return float(np.exp(np.interp(np.log(0.5 * a0), np.log(a[[k, k - 1]]), np.log(I_grid[[k, k - 1]]))))

if __name__ == "__main__":
    jobs = [(i, w) for i in range(len(T["states"])) for w in ("tw", "ad")]
    with Pool(4) as p:
        res = p.map(rates, jobs)
    Ig = np.logspace(7, 12, 400)
    for r in res:
        st = T["states"][r["i"]][r["design"]]
        m1 = pr.Medium(st, HW); m2 = pr.Medium(st, HW, R=r["R"])
        r["Isat_tau1_MWcm2"] = isat(m1, Ig) / 1e10
        r["Isat_LO_MWcm2"] = isat(m2, Ig) / 1e10
        R = np.array(r["R"]); n = len(R)
        r["lifetimes_ps"] = [float(1e12 / R[k, :k].sum()) if k > 0 else None for k in range(n)]
        print(f"ns {r['ns']:.0e} {r['design']}: dE_max vs tables {r['dE_max_meV']:.3f} meV | Isat tau=1ps {r['Isat_tau1_MWcm2']:.2f}"
              f" | Isat LO {r['Isat_LO_MWcm2']:.2f} MW/cm2 | out-lifetimes {[None if x is None else round(x,2) for x in r['lifetimes_ps']]}")
    json.dump(res, open(os.path.join(here, "rates.json"), "w"))
