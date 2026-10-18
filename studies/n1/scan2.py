"""N1 scans v2: fixed sheet density 5e11 cm^-2 per period, 200 A spacer, pump 10 um, G = 10 meV,
chi(2) at 300 K and 77 K with Fermi-Dirac subband populations; escape broadening of states 1-4."""
import numpy as np, itertools, json, sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from multiprocessing import Pool
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from qwkit import core as q
HW = 1.23984 / 10.0
DZ = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

def lay_of(fam, g):
    if fam == "adqw": return [(g[0], "well"), (g[1], "barrier"), (g[2], "well")]
    if fam == "sqw":  return [(g[0], "well")]
    if fam == "sdqw": return [(g[0], "well"), (g[1], "barrier"), (g[0], "well")]
    if fam == "stqw": return [(g[0], "well"), (g[1], "barrier"), (g[2], "well"), (g[1], "barrier"), (g[0], "well")]

def run(args):
    fam, g, F = args
    try:
        r = q.analyze_T(lay_of(fam, g), F, HW, dz=DZ)
        E = (r["E"] - r["E"][0]) * 1e3
        esc = [q.escape_broadening(r, k, F)[0] for k in range(min(4, len(E)))]
        escmax = float(max(esc[:3])) if F else 0.0
        return dict(fam=fam, g=list(g), F=F, chi300=abs(r["chiT"][300.0]), chi77=abs(r["chiT"][77.0]),
                    chi300c=[r["chiT"][300.0].real, r["chiT"][300.0].imag],
                    alpha=r["alpha"], E=[float(x) for x in E[:5]], rho2_300=float(r["rhoT"][300.0][1]),
                    esc=escmax, N=r["N_cm3"])
    except Exception as ex:
        return dict(fam=fam, g=list(g), F=F, chi300=0.0, chi77=0.0, err=str(ex))

def jobs(fam):
    Fs = [0.0] if fam == "adqw" else [float(f) for f in range(10, 210, 10)]
    if fam == "adqw": geos = itertools.product(range(40, 92, 4), range(8, 34, 4), range(20, 52, 4))
    if fam == "sqw":  geos = [(w,) for w in range(40, 136, 4)]
    if fam == "sdqw": geos = itertools.product(range(28, 84, 4), range(6, 42, 4))
    if fam == "stqw": geos = itertools.product(range(24, 72, 6), range(6, 32, 6), range(24, 84, 8))
    for g in geos:
        for F in Fs: yield (fam, g, F)

if __name__ == "__main__":
    fam = sys.argv[1]
    with Pool(2) as p:
        res = p.map(run, list(jobs(fam)), chunksize=8)
    json.dump(res, open(f"v2_{fam}.json", "w"))
    ok = [r for r in res if "err" not in r]
    ok.sort(key=lambda r: -r["chi300"])
    print(fam, len(res), "errors", len(res) - len(ok))
    for r in ok[:4]: print(r["g"], r["F"], round(r["chi300"]), round(r["chi77"]), [round(x, 1) for x in r["E"][:4]], "esc", round(r["esc"], 3), "rho2", round(r["rho2_300"], 3))
