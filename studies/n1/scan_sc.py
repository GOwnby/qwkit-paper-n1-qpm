"""Gate 1 re-check with self-consistent Hartree + depolarisation.
Stage A: coarse grid; stage B: refinement around the best. Warm-started across field values.
Usage: python3 scan_sc.py A|B  (B reads sc_A.json)"""
import sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp
HW = 1.23984 / 10; DZ = 1.0
def tw(g): return [(g[0], "well"), (g[1], "barrier"), (g[2], "well"), (g[1], "barrier"), (g[0], "well")]
def ad(g): return [(g[0], "well"), (g[1], "barrier"), (g[2], "well")]

def run_tw(args):
    g, Fs = args; out = []; VH = None
    for F in Fs:
        try:
            o = sp.chi2_sc(tw(g), float(F), HW, dz=DZ, VH0=VH); VH = o["r"]["VH"]
            out.append(dict(g=list(g), F=F, chi=abs(o["chi"]), iters=o["r"]["iters"]))
        except Exception as ex:
            out.append(dict(g=list(g), F=F, chi=0.0, err=str(ex)))
    return out
def run_ad(g):
    try:
        o = sp.chi2_sc(ad(g), 0.0, HW, dz=DZ); return dict(g=list(g), chi=abs(o["chi"]))
    except Exception as ex:
        return dict(g=list(g), chi=0.0, err=str(ex))

if __name__ == "__main__":
    stage = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    if stage == "A":
        G = list(itertools.product(range(50, 70, 4), range(10, 20, 4), range(26, 46, 4))); Fs = list(range(30, 90, 10))
        A = list(itertools.product(range(58, 72, 2), range(6, 16, 2), range(24, 36, 2)))
    else:
        prev = json.load(open(os.path.join(here, "sc_A.json")))
        bt = max(prev["tw"], key=lambda r: r["chi"])["g"]; ba = max(prev["ad"], key=lambda r: r["chi"])["g"]
        G = list(itertools.product(range(bt[0]-3, bt[0]+4, 1), range(bt[1]-2, bt[1]+3, 2), range(bt[2]-3, bt[2]+4, 1))); Fs = list(range(25, 95, 5))
        A = list(itertools.product(range(ba[0]-2, ba[0]+3, 1), range(ba[1]-2, ba[1]+3, 1), range(ba[2]-2, ba[2]+3, 1)))
    with Pool(2) as p:
        ad_res = p.map(run_ad, A)
        tw_res = [x for lst in p.map(run_tw, [(g, Fs) for g in G]) for x in lst]
    json.dump(dict(tw=tw_res, ad=ad_res), open(os.path.join(here, f"sc_{stage}.json"), "w"))
    bt = max(tw_res, key=lambda r: r["chi"]); ba = max(ad_res, key=lambda r: r["chi"])
    print(stage, "best TW", bt, "best ADQW", ba, "ratio %.3f" % (bt["chi"] / ba["chi"]), flush=True)
