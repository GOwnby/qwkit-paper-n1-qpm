"""Residual linear (absorption and index) grating written by the electrodes.
Triple well: alpha and n are even in F, so along the alternating-electrode pattern they vary only in the
gaps, with period Lambda/2 (Fourier component at the QPM wavevector K = 2 pi / Lambda vanishes by symmetry).
ADQW with Stark on/off QPM (off field from stark_valid.json): the linear response differs between 'on' (0 V) and 'off' segments, giving a
grating at K. Modal (confinement-weighted) values. Writes linear_grating.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from qwkit import sp, propagation as pr
import propagation_study as ps
here = os.path.dirname(os.path.abspath(__file__))
HW = ps.HW
TW = [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")]
AD = [(64, "well"), (10, "barrier"), (29, "well")]
G1, G2 = ps.WG["G1"], ps.WG["G2"]

def linear(lay, F, ns=5e11):
    o = sp.chi2_sc(lay, float(F), HW, ns_cm2=ns)
    st = dict(E=o["E"].tolist(), Z=o["Z"].tolist(), Et=o["Et"].tolist(), rho=o["rho"].tolist(), N=o["N"])
    med = pr.Medium(st, HW)
    a1, a2 = med.alpha(np.array(st["rho"]))
    n1 = G1 * ps.chi1_re(st, HW) / (2 * ps.WG["n1"]); n2 = G2 * ps.chi1_re(st, 2 * HW) / (2 * ps.WG["n2"])
    return dict(F=F, chi=abs(o["chi"]), a1=G1 * a1 / 100, a2=G2 * a2 / 100, dn1=n1, dn2=n2)   # alpha in 1/cm (modal)

if __name__ == "__main__":
    Ft = list(range(0, 66, 5))
    tw = [linear(TW, F) for F in Ft]
    E = json.load(open(os.path.join(here, "electrodes.json")))
    x = np.array(E["profile"]["x_um"]); Fx = np.abs(np.array(E["profile"]["F"])); Lam = E["Lam_um"]
    out = dict(tw=tw)
    def comp(key, k):
        v = np.interp(Fx, Ft, [r[key] for r in tw])
        return abs(np.mean(v * np.exp(-1j * 2 * np.pi * k * x / Lam))), v.max() - v.min()
    out["tw_fourier"] = {key: dict(K=comp(key, 1)[0], K2=comp(key, 2)[0], pp=comp(key, 1)[1]) for key in ("a1", "a2", "dn1", "dn2")}
    # ADQW Stark: off field from stark_valid.json (best valid off field at 10 W, 5e11 cm^-2)
    SV = json.load(open(os.path.join(here, "stark_valid.json")))["results"]
    Foff = min([r for r in SV if abs(r["ns"] - 5e11) < 1], key=lambda r: abs(r["P"] - 10.0))["F_off"]
    ad = [linear(AD, 0)]
    c0 = ad[0]["chi"]
    off = linear(AD, Foff)
    out["ad_on"], out["ad_off"] = ad[0], off
    out["ad_K"] = {key: abs(ad[0][key] - off[key]) / np.pi for key in ("a1", "a2", "dn1", "dn2")}   # square wave: |Δ|/π
    json.dump(out, open(os.path.join(here, "linear_grating.json"), "w"))
    print("TW alpha1/alpha2 (1/cm, modal) at 0 and 55 kV/cm:", [(r["F"], round(r["a1"], 1), round(r["a2"], 1)) for r in tw if r["F"] in (0, 55)])
    for key, v in out["tw_fourier"].items():
        print(f"TW {key}: |component at K| {v['K']:.2e}, at 2K {v['K2']:.2e}, peak-to-peak {v['pp']:.2e}")
    print(f"ADQW Stark off-field {off['F']} kV/cm (chi {off['chi']/c0*100:.0f}% of on)")
    for key in ("a1", "a2", "dn1", "dn2"):
        print(f"ADQW Stark {key}: on {ad[0][key]:.3e}, off {off[key]:.3e}, component at K {out['ad_K'][key]:.2e}")
