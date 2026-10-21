"""N1 electrical schemes with the electrode fields themselves instead of one electrode factor (LATER_STAGE 51).

The earlier N1 results multiplied chi(2) of every electrically written pattern by eta_e = 0.92, the QPM Fourier
coefficient of the triple-well (TW) sign grating under 2 um gaps. Here each pattern is propagated through the field
that the electrodes actually produce: Laplace's equation for the segmented top plane 5.31 um above the ground plane
(qwkit.electrodes.field_profile), evaluated at the bottom, centre and top of the QW stack, each height taking the
self-consistent medium (chi(2), absorption, ISB index, LO-phonon relaxation) of the nearest bias on a 5 kV/cm grid.
This is the field model of the N2 study (studies/n2/common_n2.py, copied unchanged; bias maps in studies/n2).

N1 conventions are kept: segments one coherence length of their own medium, dk from the equilibrium ISB index
(no index saturation), Gamma = 10 meV, LO-phonon rates, valid single-period fields only. New here: a lateral
breakdown limit of 0.5 MV/cm across the 2 um gaps, so neighbouring segments may differ by at most 188 kV/cm.
Abrupt ("ideal") fields are available for comparison (no electrode factor)."""
import os, sys, json
os.environ.setdefault("N2_INDEX_SAT", "0")           # N1 conventions: dk from the equilibrium ISB index
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "n2")); sys.path.insert(0, os.path.join(here, "..", "..")); sys.path.insert(0, here)
import common_n2 as c

NS = [1e10, 3e10, 1e11, 2e11, 5e11]                  # tables.json / bias-map index I = 0 ... 4
LMAX = {0: 6000.0, 1: 3500.0, 2: 1500.0, 3: 900.0, 4: 400.0}   # um, above every N1 optimum length at that density
DX = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.25, 4: 0.25}                # um
GAP = 2.0
DFM = c.dfmax(GAP)                                    # 188 kV/cm between neighbouring electrodes
_B = {}


def bank(design, I, dhw=0.0):
    k = (design, I, round(dhw, 4))
    if k not in _B:
        _B[k] = c.Bank(design, I, dhw)
    return _B[k]


def has(b, F):
    return bool(np.any(np.abs(b.F - F) < 1e-9))


def periodic(b, fields, L):
    """Periodic segments at the given fields, each one coherence length of its own medium, covering L."""
    ls = [b.Lcoh[b.index_of(F)] for F in fields]
    n = int(L / sum(ls)) + 2
    return list(fields) * n, ls * n


def ev(b, fields, P, I, real=True, record=False):
    """Peak SH efficiency and its length for a periodic pattern (realistic or abrupt fields)."""
    F, Ls = periodic(b, fields, LMAX[I])
    if real:
        idx = c.realistic_steps(b, F, Ls, DX[I], L_um=LMAX[I], gap_um=GAP)
    else:
        idx = c.ideal_steps(b, F, Ls, DX[I])[: int(round(LMAX[I] / DX[I]))]
    r = c.propagate(b, idx, DX[I], P, record=record)
    out = dict(eta=float(r["eta"]), L=float(r["L_opt_um"]), at_end=bool(r["L_opt_um"] > 0.97 * LMAX[I]))
    if record:
        out["x_um"] = r["x_um"][::8].tolist(); out["P2"] = r["P2"][::8].tolist()
    return out


def allowed(F1, F2):
    return abs(F1 - F2) <= DFM + 1e-9


def run_jobs(fn, jobs, tag, nproc=None):
    """Resumable runner: each finished job is kept in parts_realfield/<tag>/<key>.json and skipped on a rerun."""
    from multiprocessing import Pool
    d = os.path.join(here, "parts_realfield", tag); os.makedirs(d, exist_ok=True)
    todo = [a for a in jobs if not os.path.exists(os.path.join(d, _key(a) + ".json"))]
    if todo:
        with Pool(nproc or int(os.environ.get("NPROC", "2"))) as pool:
            for a, r in zip(todo, pool.imap(_Call(fn, d), todo, chunksize=1)):
                pass
    return [json.load(open(os.path.join(d, _key(a) + ".json"))) for a in jobs]


def _key(a):
    return "_".join(str(x) for x in a)


class _Call:
    def __init__(self, fn, d):
        self.fn, self.d = fn, d

    def __call__(self, a):
        r = self.fn(a)
        p = os.path.join(self.d, _key(a) + ".json")
        json.dump(r, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
        return 0
