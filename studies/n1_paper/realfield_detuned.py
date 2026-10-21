"""N1 Table 4 (each scheme at its own best red detuning) for the electrical schemes with the electrode fields
(realfield_common.py): triple-well sign reversal at +-55 kV/cm and Stark on/off QPM (0 V and the best valid off
field, |F| = 10-140 kV/cm in 5 kV/cm steps, re-chosen at each pump energy), pump detunings 0 to -40 meV as in
detuned_qpm.py (waveguide recomputed), P = 1.26, 4 and 10 W, ns = 1e11 and 5e11 cm^-2. Growth reversal and
phase matching involve no electrodes and stay as in detuned_qpm*.json. Red-detuned optima are longer than resonant
ones, so the guide is 3 mm (1e11) and 0.8 mm (5e11 cm^-2) long here, beyond the growth-reversal optima of
detuned_qpm*.json. Writes realfield_detuned.json."""
import os, sys, json, time
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import realfield_common as rc
rc.LMAX.update({2: 3000.0, 4: 800.0})

DHW = [0.0, -0.01, -0.02, -0.025, -0.03, -0.035, -0.04]
PP = [1.26, 4.0, 10.0]


def job(a):
    I, dhw = a; t0 = time.time()
    tw, ad = rc.bank("tw", I, dhw), rc.bank("ad", I, dhw)
    out = []
    for P in PP:
        t = rc.ev(tw, [55, -55], P, I)
        cand = [F for F in range(-140, 145, 5) if abs(F) >= 10 and rc.has(ad, F)]
        r = {F: rc.ev(ad, [0, F], P, I)["eta"] for F in cand}
        Fb = max(r, key=r.get)
        s = rc.ev(ad, [0, Fb], P, I)
        out.append(dict(I=I, ns=rc.NS[I], dhw=dhw, P=P, tw55=t["eta"], tw55_L=t["L"], tw55_at_end=t["at_end"], stark=r[Fb], F_off=Fb,
                        stark_L=s["L"], stark_at_end=s["at_end"]))
        print(f"ns {rc.NS[I]:.0e} dhw {1e3*dhw:+.0f} P {P}: TW55 {100*t['eta']:.4f}% | Stark {100*r[Fb]:.4f}% ({Fb:+d})", flush=True)
    return out


if __name__ == "__main__":
    jobs = [(I, dhw) for I in (4, 2) for dhw in DHW]
    res = [x for r in rc.run_jobs(job, jobs, "detuned") for x in r]
    json.dump(res, open(os.path.join(here, "realfield_detuned.json"), "w"))
