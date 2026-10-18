"""Companion to twofield_qpm.py: the triple well with BOTH segment fields optimized (general pairs F1, F2 from
F in {+-20, ..., +-120} kV/cm where the single-period model is valid), so that both structures get the same freedom.
Electrode factor applied to both segments as in the sign-reversal case. Writes twofield_tw.json."""
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
import detuned_qpm as dq
import twofield_qpm as tq
here = os.path.dirname(os.path.abspath(__file__))
FT = [s * f for f in range(20, 140, 20) for s in (1, -1)]
DHW = [0.0, -0.010, -0.020]
PS = [1.26, 10.0]


def run(args):
    I, a, b, dhw, P = args
    hw = ps.HW + dhw; base = dq.wg_at(dhw)
    m1 = pr.Medium(a["st"], hw, gamma=0.010, R=a["R"]); m2 = pr.Medium(b["st"], hw, gamma=0.010, R=b["R"])
    w1 = dq.wg_for(a["st"], hw, base); w2 = dq.wg_for(b["st"], hw, base)
    r = pr.propagate_segments([(m1, w1["Lcoh_um"], ps.ETA_E), (m2, w2["Lcoh_um"], ps.ETA_E)], base, P, L_max_um=6000.0)
    return dict(I=I, F1=a["F"], F2=b["F"], dhw=dhw, P=P, eta=float(r["eta"]), L=float(r["L_opt_um"]))


if __name__ == "__main__":
    with Pool(2) as pool:
        sts = pool.map(tq.state, [("tw", I, F) for I in tq.NSI for F in FT])
    jobs = []
    for I in tq.NSI:
        tw = [s for s in sts if s["I"] == I and s["valid"]]
        print(f"ns_i {I}: valid TW fields {[s['F'] for s in tw]}", flush=True)
        for a, b in itertools.combinations(tw, 2):
            jobs += [(I, a, b, d, P) for d in DHW for P in PS]
    print(len(jobs), "propagations", flush=True)
    with Pool(2) as pool:
        res = pool.map(run, jobs, chunksize=8)
    json.dump(dict(results=res, valid={f"tw_{s['I']}_{s['F']}": s["valid"] for s in sts}), open(os.path.join(here, "twofield_tw.json"), "w"), indent=1)
    for I in tq.NSI:
        for P in PS:
            line = []
            for d in DHW:
                b = max([r for r in res if r["I"] == I and r["P"] == P and r["dhw"] == d], key=lambda r: r["eta"])
                line.append(f"{d*1e3:+.0f}: {b['eta']*100:.3f}% ({b['F1']:+d},{b['F2']:+d})")
            print(f"ns_i {I} P {P} tw-general: " + " | ".join(line), flush=True)
