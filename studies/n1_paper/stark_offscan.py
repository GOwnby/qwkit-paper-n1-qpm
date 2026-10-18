"""N1 benchmark revision: realistic Stark on/off QPM with the off field chosen to MAXIMIZE the Stark efficiency
(both polarities, |F| = 20-240 kV/cm), restricted to fields where the single-period model is valid, i.e. no subband
below E1 + 0.40 eV is partially localized (active-region weight between 0.15 and 0.8; such states are hybrids with
states of the field-tilted spacer, which in a real multiple-quantum-well stack belong to the neighbouring periods).
For every field we also record the number of such hybrid states and |chi2_off/chi2_on| with the standard (0.8) and a
permissive (0.15) localisation threshold. Resonant (dhw = 0) and red-detuned pumps, ns = 1e11 and 5e11 cm^-2,
P = 1.26, 4 and 10 W. Writes stark_offscan.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, core as q, scattering as sc, propagation as pr
import propagation_study as ps
import detuned_qpm as dq
here = os.path.dirname(os.path.abspath(__file__))
AD = [(64, "well"), (10, "barrier"), (29, "well")]
FS = [s * f for f in range(20, 250, 20) for s in (1, -1)]
DHW = [0.0, -0.010, -0.020, -0.025, -0.030, -0.035]
PS = [1.26, 4.0, 10.0]
WIN = 0.40


def off_state(args):
    I, F = args
    ns = ps.T["states"][I]["ns"]
    r = sp.self_consistent(AD, float(F), ns, maxit=400, nstates=10)
    z, psi, act = r["z"], r["psi"], r["active"]; dz = z[1] - z[0]
    sel = (z >= act[0] - 25) & (z <= act[1] + 25)
    w = np.sum(psi[sel] ** 2, axis=0) * dz
    inwin = (r["E"] - r["E"][0]) < WIN
    n_hyb = int(np.sum(inwin & (w > 0.15) & (w < 0.8)))
    E, psi2, Z, rho = sp.prepare(r)
    N = ns / ((act[1] - act[0] + 200) * 1e-8)
    Et = sp.depol_energies(z, psi2, E, rho, ns, eps_r=13.5)
    st = dict(E=E.tolist(), Z=Z.tolist(), Et=Et.tolist(), rho=rho.tolist(), N=N)
    return dict(I=I, F=F, valid=n_hyb == 0, n_hyb=n_hyb, weights=w[inwin].tolist(), st=st, R=sc.rate_matrix(z, psi2, E).tolist())


def prop(args):
    I, F, dhw, P, off = args
    os.environ["N1_NS_INDEX"] = str(I)
    dq.I5 = I; dq.NS = ps.T["states"][I]["ns"]
    return dict(I=I, F=F, dhw=dhw, P=P, eta=dq.job(("ad_stark", P, dhw, off))["eta"])


if __name__ == "__main__":
    with Pool(2) as pool:
        offs = pool.map(off_state, [(I, F) for I in (2, 4) for F in FS])
    for o in offs:
        on = ps.T["states"][o["I"]]["ad"]
        o["chi_ratio"] = {f"{d:.3f}": dq.chi_abs(o["st"], ps.HW + d) / dq.chi_abs(on, ps.HW + d) for d in DHW}
        print(f"ns_i {o['I']} F {o['F']:+4d}: valid {o['valid']} hybrids {o['n_hyb']} |chi_off/chi_on| at resonance {o['chi_ratio']['0.000']:.2f}", flush=True)
    jobs = [(o["I"], o["F"], d, P, o) for o in offs if o["valid"] for d in DHW for P in PS]
    with Pool(2) as pool:
        res = pool.map(prop, jobs, chunksize=4)
    json.dump(dict(offs=[{k: v for k, v in o.items() if k not in ("st", "R")} for o in offs], results=res),
              open(os.path.join(here, "stark_offscan.json"), "w"), indent=1)
    for I in (2, 4):
        for P in PS:
            line = []
            for d in DHW:
                rr = [r for r in res if r["I"] == I and r["P"] == P and r["dhw"] == d]
                b = max(rr, key=lambda r: r["eta"])
                line.append(f"{d*1e3:+.0f}: {b['eta']*100:.3f}% (F {b['F']:+d})")
            print(f"ns_i {I} P {P}: " + " | ".join(line), flush=True)
