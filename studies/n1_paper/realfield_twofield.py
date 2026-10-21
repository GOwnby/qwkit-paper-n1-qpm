"""N1 Table 5 (optimized two-field patterns) with the electrode fields (realfield_common.py): every ordered pair
(F1, F2) of valid fields on a 10 kV/cm grid, |F| <= 120 kV/cm (ADQW) or <= 100 kV/cm (triple well, the range of its
bias map), within the 188 kV/cm breakdown limit; segments one coherence length each; pump detuning 0, -10, -20,
-30 meV (waveguide recomputed); P = 1.26 and 10 W; ns = 1e11 and 5e11 cm^-2; LO-phonon relaxation.
Writes realfield_twofield.json (all pairs and the best per case)."""
import os, sys, json, time, itertools
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import realfield_common as rc

DHW = [0.0, -0.01, -0.02, -0.03]
PP = [1.26, 10.0]


def job(a):
    design, I, dhw, P = a; t0 = time.time()
    b = rc.bank(design, I, dhw)
    fmax = 120 if design == "ad" else 100
    grid = [F for F in range(-fmax, fmax + 10, 10) if rc.has(b, F)]
    pairs = [(x, y) for x, y in itertools.permutations(grid, 2) if rc.allowed(x, y)]
    res = []
    for x, y in pairs:
        r = rc.ev(b, [x, y], P, I)
        res.append(dict(F1=x, F2=y, eta=r["eta"], L=r["L"], at_end=r["at_end"]))
    best = max(res, key=lambda r: r["eta"])
    sym = [r for r in res if r["F1"] == -r["F2"]]
    print(f"{design} ns {rc.NS[I]:.0e} dhw {1e3*dhw:+.0f} P {P}: best ({best['F1']:+d},{best['F2']:+d}) {100*best['eta']:.4f}% "
          f"| best +-F {100*max(r['eta'] for r in sym):.4f}% | {len(pairs)} pairs {time.time()-t0:.0f} s", flush=True)
    return dict(design=design, I=I, ns=rc.NS[I], dhw=dhw, P=P, best=best, best_sym=max(sym, key=lambda r: r["eta"]), all=res)


if __name__ == "__main__":
    jobs = [(d, I, dhw, P) for I in (4, 2) for d in ("ad", "tw") for dhw in DHW for P in PP]
    res = rc.run_jobs(job, jobs, "twofield")
    json.dump(res, open(os.path.join(here, "realfield_twofield.json"), "w"))
