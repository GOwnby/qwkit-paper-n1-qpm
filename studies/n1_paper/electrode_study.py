"""Effective QPM coefficient of the triple well under real alternating electrodes.
Uses the self-consistent chi(F) table (tables.json) and the nominal waveguide (waveguide.json)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from qwkit import electrodes as el
here = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(here, "tables.json")))
W = [w for w in json.load(open(os.path.join(here, "waveguide.json"))) if w["t_up"] == 2.0 and w["t_sp"] == 0.9][0]
Fs = np.array([r["F"] for r in T["chiF"]]); chis = np.array([r["re"] + 1j * r["im"] for r in T["chiF"]])
chimax = np.abs(chis).max()
def chi_of(F):                                   # F in kV/cm, signed; chi odd in F
    a = np.abs(F); c = np.interp(a, Fs, chis.real) + 1j * np.interp(a, Fs, chis.imag)
    c = np.where(a > Fs.max(), np.interp(Fs.max(), Fs, chis.real), c)
    return np.sign(F) * c
d = (W["t_up"] + 2 * W["t_sp"] + 40 * 378e-4) * 1e-6            # electrode plane to ground (m)
y_lo, y_hi = W["t_sp"] * 1e-6, (W["t_sp"] + 40 * 378e-4) * 1e-6  # stack height range above ground
Lam = 2 * W["Lcoh_um"] * 1e-6
ys = np.linspace(y_lo, y_hi, 7)
res = []
for gap_um in (1.0, 2.0, 4.0):
    best = None
    for V in np.arange(10, 61, 1.0):
        cs = []
        for y in ys:
            x, Ey = el.field_map(Lam, d, V, gap_um * 1e-6, y, nx=2048, nmax=301)
            cs.append(chi_of(Ey / 1e5))              # V/m -> kV/cm
        chi_avg = np.mean(cs, axis=0)
        c1 = el.qpm_coefficient(x, chi_avg, Lam)
        if best is None or c1 > best["c1"]:
            best = dict(gap_um=gap_um, V=float(V), c1=float(c1), rel_ideal=float(c1 / (2 / np.pi * chimax)),
                        Fplateau=float(np.abs(Ey).max() / 1e5))
    res.append(best)
    print(f"gap {gap_um} um: best V {best['V']:.0f} V, plateau field {best['Fplateau']:.0f} kV/cm, "
          f"QPM coefficient {best['rel_ideal']*100:.1f}% of ideal sign reversal (2/pi x max|chi|)")
# field profile for the figure at the best V for gap 2 um, stack centre
b = [r for r in res if r["gap_um"] == 2.0][0]
x, Ey = el.field_map(Lam, d, b["V"], 2e-6, 0.5 * (y_lo + y_hi), nx=1024, nmax=301)
json.dump(dict(results=res, Lam_um=Lam * 1e6, d_um=d * 1e6, profile=dict(x_um=(x * 1e6).tolist(), F=(Ey / 1e5).tolist(),
          chi=(np.abs(chi_of(Ey / 1e5)) * np.sign(Ey) / chimax).tolist())), open(os.path.join(here, "electrodes.json"), "w"))
