"""Summary of the two-field optimizations (twofield_qpm.json, twofield_tw.json) for the N1 manuscript, with the best
ADQW patterns re-run with the same electrode factor as the triple well. Writes twofield_best.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
import detuned_qpm as dq
import twofield_qpm as tq
here = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(here, "twofield_qpm.json")))["results"]
B = json.load(open(os.path.join(here, "twofield_tw.json")))["results"]


def rerun(r):
    a = tq.state(("ad", r["I"], r["F1"])); b = tq.state(("ad", r["I"], r["F2"]))
    hw = ps.HW + r["dhw"]; base = dq.wg_at(r["dhw"])
    m1 = pr.Medium(a["st"], hw, gamma=0.010, R=a["R"]); m2 = pr.Medium(b["st"], hw, gamma=0.010, R=b["R"])
    w1 = dq.wg_for(a["st"], hw, base); w2 = dq.wg_for(b["st"], hw, base)
    x = pr.propagate_segments([(m1, w1["Lcoh_um"], ps.ETA_E), (m2, w2["Lcoh_um"], ps.ETA_E)], base, r["P"], L_max_um=6000.0)
    return dict(r, eta_e_applied=float(x["eta"]), Lcoh=[w1["Lcoh_um"], w2["Lcoh_um"]])


if __name__ == "__main__":
    rows = []
    for I in (2, 4):
        for P in (1.26, 10.0):
            ad = [r for r in A if r["I"] == I and r["P"] == P and r["kind"] == "ad"]
            tw = [r for r in A if r["I"] == I and r["P"] == P and r["kind"] == "tw"] + [r for r in B if r["I"] == I and r["P"] == P]
            rows.append(dict(I=I, ns=ps.T["states"][I]["ns"], P=P, ad_res=max([r for r in ad if r["dhw"] == 0], key=lambda r: r["eta"]),
                             ad_best=max(ad, key=lambda r: r["eta"]), tw_res=max([r for r in tw if r["dhw"] == 0], key=lambda r: r["eta"]),
                             tw_best=max(tw, key=lambda r: r["eta"])))
    with Pool(2) as pool:
        re = pool.map(rerun, [r["ad_best"] for r in rows])
    for r, x in zip(rows, re):
        r["ad_best"] = x
        print(f"ns {r['ns']:.0e} P {r['P']}: ADQW best {x['eta']*100:.3f}% (eta_e applied {x['eta_e_applied']*100:.3f}%) at ({x['F1']:+d},{x['F2']:+d}) dhw {x['dhw']*1e3:+.0f}, "
              f"Lcoh {x['Lcoh'][0]:.1f}/{x['Lcoh'][1]:.1f} um | TW best {r['tw_best']['eta']*100:.3f}% at ({r['tw_best']['F1']:+d},{r['tw_best']['F2']:+d}) dhw {r['tw_best']['dhw']*1e3:+.0f}"
              f" | ADQW/TW {x['eta_e_applied']/r['tw_best']['eta']:.2f}", flush=True)
    json.dump(rows, open(os.path.join(here, "twofield_best.json"), "w"), indent=1)
