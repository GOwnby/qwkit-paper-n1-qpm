"""Collects the N1 numbers that depend on the electrode fields (realfield_eff/twofield/detuned/sens.json) together with
the benchmarks that involve no electrodes (propagation_lo.json, detuned_qpm*.json, sensitivity.json), writes the
LaTeX rows of Tables 3-6 (realfield_tables.tex) and realfield_summary.json, and prints every number quoted in the
text."""
import os, sys, json
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import realfield_common as rc
import common_n2 as c

J = lambda f: json.load(open(os.path.join(here, f)))
EFF = J("realfield_eff.json"); TWO = J("realfield_twofield.json"); DET = J("realfield_detuned.json"); SEN = J("realfield_sens.json")
LO = J("propagation_lo.json")["results"]
out = {}; lines = []
say = lambda s: (lines.append(s), print(s))
PS = sorted(set(r["P"] for r in EFF))


def lo(ns, P, d, s, key="eta"):
    return [r for r in LO if abs(r["ns"] - ns) < 1 and abs(r["P"] - P) < 1e-6 and r["design"] == d and r["scheme"] == s][0][key]


def eff(ns, P):
    return [r for r in EFF if abs(r["ns"] - ns) < 1 and abs(r["P"] - P) < 1e-6][0]


def val(e, k):
    return e[k]["real"]["eta"] if k == "tw55" else e[k]["eta"]


def dm(x):
    return "0" if x == 0 else f"${1e3*x:.0f}$"


def fmt(x, nd=3):
    return f"{100*x:.{nd}f}"


def flen(L):
    return f"{L/1000:.2f} mm" if L >= 1000 else f"{L:.0f} $\\mu$m"


# ---- Table 3 (resonant pump) ----
rows = []
for P in (PS[7], PS[10]):
    for ns in (1e10, 1e11, 5e11):
        e = eff(ns, P)
        none = lo(ns, P, "ad", "none")
        nstr = "$<10^{-4}$" if none < 1e-6 else (f"${none*100:.0e}$".replace("e-0", "\\times10^{-").replace("$", "$") if none < 1e-3 else fmt(none))
        if none < 1e-6:
            nstr = "$<10^{-4}$"
        elif none < 1e-5:
            nstr = f"${none*1e6:.0f}\\times10^{{-4}}$"
        else:
            nstr = fmt(none)
        nsl = {1e10: "$10^{10}$", 1e11: "$10^{11}$", 5e11: "$5\\times10^{11}$"}[ns]
        rows.append(f"{(f'{P:.2f}' if P < 5 else f'{P:.1f}')} W & {nsl} & {fmt(e['tw55']['real']['eta'])} & {fmt(e['twopt']['eta'])} & {fmt(e['adbip']['eta'])} & "
                    f"{fmt(e['stark']['eta'])} & {fmt(lo(ns, P, 'ad', 'onoff'))} & {fmt(lo(ns, P, 'ad', 'sign'))} & {nstr} & "
                    f"{fmt(lo(ns, P, 'ad', 'perfect'))} & {flen(e['tw55']['real']['L'])} \\\\")
out["table3"] = rows
say("Table 3 rows:"); [say("  " + r) for r in rows]
flags = [(r["ns"], r["P"], k) for r in EFF for k in ("tw55", "twopt", "adbip", "stark") if r[k].get("real", {}).get("at_end")]
say(f"optimum at the end of the computed length: {flags}")

# ---- ratios (all dopings) ----
NSALL = sorted(set(r["ns"] for r in EFF))
def ratio(fn, Pcond):
    v = [fn(eff(ns, P), ns, P) for ns in NSALL for P in PS if Pcond(P)]
    return min(v), max(v)
low = lambda P: P < 0.011
R = {}
R["low_twopt_adbip"] = ratio(lambda e, ns, P: e["twopt"]["eta"] / e["adbip"]["eta"], low)
R["low_twopt_stark"] = ratio(lambda e, ns, P: e["twopt"]["eta"] / e["stark"]["eta"], low)
R["low_twopt_rev"] = ratio(lambda e, ns, P: e["twopt"]["eta"] / lo(ns, P, "ad", "sign"), low)
R["low_all_spread"] = ratio(lambda e, ns, P: max(val(e, k) for k in ("tw55", "twopt", "adbip", "stark")) /
                           min(val(e, k) for k in ("tw55", "twopt", "adbip", "stark")), low)
R["twopt_stark_1W"] = ratio(lambda e, ns, P: e["twopt"]["eta"] / e["stark"]["eta"], lambda P: abs(P - PS[7]) < 1e-6)
R["twopt_stark_10W"] = ratio(lambda e, ns, P: e["twopt"]["eta"] / e["stark"]["eta"], lambda P: abs(P - PS[10]) < 1e-6)
R["twopt_rev_1to20W"] = ratio(lambda e, ns, P: e["twopt"]["eta"] / lo(ns, P, "ad", "sign"), lambda P: P > 1.2)
R["twopt_over_tw55_10W"] = ratio(lambda e, ns, P: e["twopt"]["eta"] / e["tw55"]["real"]["eta"], lambda P: abs(P - PS[10]) < 1e-6)
R["adbip_twopt_above1W"] = ratio(lambda e, ns, P: e["adbip"]["eta"] / e["twopt"]["eta"], lambda P: P > 1.2)
R["adbip_best_scheme_above1W"] = ratio(lambda e, ns, P: e["adbip"]["eta"] / max(val(e, k) for k in ("tw55", "twopt", "stark")), lambda P: P > 1.2)
R["tw55_real_over_ideal"] = ratio(lambda e, ns, P: e["tw55"]["real"]["eta"] / e["tw55"]["ideal"]["eta"], lambda P: P > 1.2)
R["adbip_real_over_ideal"] = ratio(lambda e, ns, P: e["adbip"]["real"]["eta"] / e["adbip"]["ideal"]["eta"], lambda P: P > 1.2)
R["stark_real_over_ideal"] = ratio(lambda e, ns, P: e["stark"]["real"]["eta"] / e["stark"]["ideal"]["eta"], lambda P: P > 1.2)
for k, v in R.items():
    say(f"{k}: {v[0]:.2f}-{v[1]:.2f}")
out["ratios"] = R
say(f"ADQW bip at 10 W, 5e11: {fmt(eff(5e11, PS[10])['adbip']['eta'])}% (+-{eff(5e11, PS[10])['adbip']['F']}); PM {fmt(lo(5e11, PS[10], 'ad', 'perfect'))}%")
Fop = lambda k: sorted(set(abs(r[k]["F"]) for r in EFF if r["P"] > 1.2))
say(f"optimum fields at P >= 1.26 W: TW opt {Fop('twopt')}, ADQW bip {Fop('adbip')}, Stark off {sorted(set(r['stark']['F'] for r in EFF if r['P'] > 1.2))}")
say(f"Stark off fields at 5e11 (all powers): {[eff(5e11, P)['stark']['F'] for P in PS]}")
out["fields"] = dict(twopt=Fop("twopt"), adbip=Fop("adbip"))

# bipolar phases and magnitudes at the optimum fields (small signal, at the pump), P >= 1.26 W
ph = []
for r in EFF:
    if r["P"] < 1.2:
        continue
    I = rc.NS.index(r["ns"]); b = rc.bank("ad", I); F = r["adbip"]["F"]
    cp, cm, c0 = b.chi0[b.index_of(F)], b.chi0[b.index_of(-F)], b.chi0[b.index_of(0)]
    d = abs(np.degrees(np.angle(cp) - np.angle(cm))) % 360
    ph.append((d if d <= 180 else 360 - d, abs(cp) / abs(c0), abs(cm) / abs(c0), F, r["ns"]))
say(f"bipolar optimum: phase difference {min(p[0] for p in ph):.0f}-{max(p[0] for p in ph):.0f} deg (as |delta| <= 180), "
    f"magnitudes {100*min(min(p[1], p[2]) for p in ph):.0f}-{100*max(max(p[1], p[2]) for p in ph):.0f}% of 0 V")
for r in EFF:
    if r["P"] > 1.2:
        I = rc.NS.index(r["ns"]); st = {s["F"]: s["st"] for s in c.statesets("ad", I)}
        F = r["adbip"]["F"]
        dE = [1e3 * (st[f]["Et"][1][0] - c.HW0) for f in (F, -F)]; dE3 = [1e3 * (st[f]["Et"][2][0] - 2 * c.HW0) for f in (F, -F)]
        r["_det"] = (dE, dE3)
say("bipolar optimum detunings E21-hw (meV) and E31-2hw: " + "; ".join(f"{r['ns']:.0e} {r['P']:.1f} W +-{r['adbip']['F']}: "
    f"{r['_det'][0][0]:+.0f}/{r['_det'][0][1]:+.0f}, {r['_det'][1][0]:+.0f}/{r['_det'][1][1]:+.0f}" for r in EFF if "_det" in r and abs(r["P"] - PS[10]) < 1e-6))
# Stark example: modal pump absorption and |chi2| at the 10 W, 5e11 optimum off field
b = rc.bank("ad", 4); Fo = eff(5e11, PS[10])["stark"]["F"]
a = lambda F: b.media[b.index_of(F)].alpha(b.media[b.index_of(F)].rho0)[0] / 100 * b.wg["G1"]
say(f"Stark 10 W 5e11 off {Fo:+d}: modal pump absorption {a(0):.0f} -> {a(Fo):.0f} cm-1, |chi2| {100*abs(b.chi0[b.index_of(Fo)])/abs(b.chi0[b.index_of(0)]):.0f}% of 0 V; "
    f"Lcoh on {b.Lcoh[b.index_of(0)]:.1f} off {b.Lcoh[b.index_of(Fo)]:.1f} um")
# Stark on/off coherence-length contrast at the optimum off fields (P >= 1.26 W)
con = []
for r in EFF:
    if r["P"] > 1.2:
        b_ = rc.bank("ad", rc.NS.index(r["ns"])); L0, L1 = b_.Lcoh[b_.index_of(0)], b_.Lcoh[b_.index_of(r["stark"]["F"])]
        con.append(abs(L1 - L0) / min(L0, L1))
say(f"Stark on/off coherence-length contrast above 1 W: {100*min(con):.0f}-{100*max(con):.0f}%")
# crossover pump power TW55 (realistic) vs growth reversal
cr = {}
for ns in NSALL:
    cr[f"{ns:.0e}"] = next((P for P in PS if eff(ns, P)["tw55"]["real"]["eta"] > lo(ns, P, "ad", "sign")), None)
say(f"TW55 > rev from P (W): {cr}; x 1.74 MW/cm2 per W")
crop = {f"{ns:.0e}": next((P for P in PS if eff(ns, P)["twopt"]["eta"] > lo(ns, P, "ad", "sign")), None) for ns in NSALL}
say(f"TW opt > rev from P (W): {crop}")
out["crossover"] = dict(tw55=cr, twopt=crop)
# heating at 10 W, 5e11, TW 55 kV/cm with the electrode fields
tw = rc.bank("tw", 4)
F, Ls = rc.periodic(tw, [55, -55], rc.LMAX[4])
rr = c.propagate(tw, c.realistic_steps(tw, F, Ls, rc.DX[4], L_um=rc.LMAX[4]), rc.DX[4], PS[10], record=True)
k = int(np.argmax(rr["P2"])); L = rr["x_um"][k] * 1e-4
Pabs = PS[10] - rr["P1"][k] - rr["P2"][k]; V = 40 * 378e-8 * 15e-4 * L; q = Pabs / V
say(f"heating TW55 10 W 5e11: L {L*1e4:.0f} um, absorbed {Pabs:.1f} W, {q:.1e} W/cm3, {q/1.5*1e-9:.1f} K/ns")
out["heating"] = dict(L_um=L * 1e4, Pabs=Pabs, q=q)

# ---- Table 5 (two-field patterns) ----
rows5 = []
for ns in (1e11, 5e11):
    for P in (1.26, 10.0):
        best = {}
        for d in ("ad", "tw"):
            rr = [r for r in TWO if r["design"] == d and abs(r["ns"] - ns) < 1 and r["P"] == P]
            bb = max(rr, key=lambda r: r["best"]["eta"])
            sym = max(rr, key=lambda r: r["best_sym"]["eta"])
            best[d] = (bb, sym)
        (ab, asym), (tb, tsym) = best["ad"], best["tw"]
        pat = lambda r: (f"$(\\pm{abs(r['F1'])})$" if r["F1"] == -r["F2"] else f"$({r['F1']:+d},{r['F2']:+d})$")
        nsl = {1e11: "$10^{11}$", 5e11: "$5\\times10^{11}$"}[ns]
        rows5.append(f"{nsl} & {P:g} W & {pat(ab['best'])}, {dm(ab['dhw'])} & {fmt(ab['best']['eta'])} & "
                     f"{pat(tb['best'])}, {dm(tb['dhw'])} & {fmt(tb['best']['eta'])} \\\\")
        Ia = rc.NS.index(ns); ba = rc.bank("ad", Ia, ab["dhw"])
        say(f"two-field {ns:.0e} {P} W: ADQW best ({ab['best']['F1']:+d},{ab['best']['F2']:+d}) dhw {1e3*ab['dhw']:.0f} {fmt(ab['best']['eta'])}% "
            f"[Lcoh {ba.Lcoh[ba.index_of(ab['best']['F1'])]:.1f}/{ba.Lcoh[ba.index_of(ab['best']['F2'])]:.1f}], best +-F {pat(asym['best_sym'])} "
            f"dhw {1e3*asym['dhw']:.0f} {fmt(asym['best_sym']['eta'])}% ({asym['best_sym']['eta']/ab['best']['eta']:.2f} of best) | "
            f"TW best ({tb['best']['F1']:+d},{tb['best']['F2']:+d}) dhw {1e3*tb['dhw']:.0f} {fmt(tb['best']['eta'])}%, best +-F "
            f"{fmt(tsym['best_sym']['eta'])}% | ADQW/TW {ab['best']['eta']/tb['best']['eta']:.2f}")
        top = sorted([x for r in TWO if r["design"] == "ad" and abs(r["ns"] - ns) < 1 and r["P"] == P and r["dhw"] == ab["dhw"] for x in r["all"]],
                     key=lambda x: -x["eta"])[:6]
        say("    ADQW top pairs: " + ", ".join(f"({x['F1']:+d},{x['F2']:+d}) {fmt(x['eta'])}" for x in top))
out["table5"] = rows5
say("Table 5 rows:"); [say("  " + r) for r in rows5]

# ---- Table 4 (each scheme at its own best detuning) ----
rows4 = []
for ns, f in ((1e11, "detuned_qpm_ns2.json"), (5e11, "detuned_qpm.json")):
    D = J(f)["results"]
    for P in (1.26, 4.0, 10.0):
        t = max([r for r in DET if abs(r["ns"] - ns) < 1 and r["P"] == P], key=lambda r: r["tw55"])
        s = max([r for r in DET if abs(r["ns"] - ns) < 1 and r["P"] == P and r["dhw"] >= -0.0351], key=lambda r: r["stark"])
        rv = max([r for r in D if r["scheme"] == "ad_rev" and r["P"] == P], key=lambda r: r["eta"])
        pm = max([r for r in D if r["scheme"] == "ad_pm" and r["P"] == P], key=lambda r: r["eta"])
        nsl = {1e11: "$10^{11}$", 5e11: "$5\\times10^{11}$"}[ns]
        d = lambda x: f"${1e3*x:.0f}$".replace("$-0$", "$0$") if x != 0 else "$0$"
        rows4.append(f"{nsl} & {P:g} W & {fmt(t['tw55'])} ({d(t['dhw'])}) & {fmt(s['stark'])} ({d(s['dhw'])}) & {fmt(rv['eta'])} ({d(rv['dhw'])}) & "
                     f"{fmt(pm['eta'])} ({d(pm['dhw'])}) & {t['tw55']/s['stark']:.2f} & {t['tw55']/rv['eta']:.2f} \\\\")
        say(f"detuned {ns:.0e} {P} W: TW55 {fmt(t['tw55'])} ({1e3*t['dhw']:.0f}) Stark {fmt(s['stark'])} ({1e3*s['dhw']:.0f}, off {s['F_off']:+d}) "
            f"rev {fmt(rv['eta'])} PM {fmt(pm['eta'])} | TW/Stark {t['tw55']/s['stark']:.2f} TW/rev {t['tw55']/rv['eta']:.2f}")
        z = [r for r in DET if abs(r["ns"] - ns) < 1 and r["P"] == P and r["dhw"] == 0.0][0]
        if P in (1.26, 10.0):
            e = eff(ns, PS[7] if P == 1.26 else PS[10])
            say(f"    resonant check: TW55 {fmt(z['tw55'])} vs Table 3 {fmt(e['tw55']['real']['eta'])}; Stark {fmt(z['stark'])} vs {fmt(e['stark']['eta'])}")
say(f"detuned optima at the end of the guide: {[(r['ns'], r['P'], r['dhw'], k) for r in DET for k in ('tw55', 'stark') if r.get(k + '_at_end')]}")
for ns in (1e11, 5e11):
    for P in (1.26, 10.0):
        tbest = max(max(x["best"]["eta"] for x in TWO if x["design"] == "tw" and abs(x["ns"] - ns) < 1 and x["P"] == P),
                    max(r["tw55"] for r in DET if abs(r["ns"] - ns) < 1 and r["P"] == P))
        abest = max(x["best"]["eta"] for x in TWO if x["design"] == "ad" and abs(x["ns"] - ns) < 1 and x["P"] == P)
        say(f"ADQW best two-field / best triple-well point (Tables 4-5) {ns:.0e} {P} W: {abest/tbest:.2f}")
out["table4"] = rows4
say("Table 4 rows:"); [say("  " + r) for r in rows4]

# ---- Table 6 (sensitivity) ----
rows6 = []
for ns in (1e11, 5e11):
    for P in (0.01, 1.26, 10.0):
        rr = [r for r in SEN if abs(r["ns"] - ns) < 1 and r["P"] == P]
        a = [r["tw"] / r["rev"] for r in rr]; b2 = [r["tw"] / r["stark"] for r in rr]; e = [r["tw"] for r in rr]
        nsl = {1e11: "$10^{11}$", 5e11: "$5\\times10^{11}$"}[ns]
        Pl = "10 mW" if P == 0.01 else f"{P:g} W"
        rows6.append(f"{nsl} & {Pl} & {min(a):.2f}--{max(a):.2f} & {min(b2):.2f}--{max(b2):.2f} & "
                     + (f"{100*min(e):.3f}--{100*max(e):.3f}" if P == 0.01 else f"{100*min(e):.2f}--{100*max(e):.2f}") + " \\\\")
out["table6"] = rows6
say("Table 6 rows:"); [say("  " + r) for r in rows6]
open(os.path.join(here, "realfield_tables.tex"), "w").write("% generated by realfield_summary.py\n" + "\n".join(
    ["% Table 3"] + rows + ["% Table 4"] + rows4 + ["% Table 5"] + rows5 + ["% Table 6"] + rows6) + "\n")
json.dump(out, open(os.path.join(here, "realfield_summary.json"), "w"), indent=1, default=str)
open(os.path.join(here, "realfield_summary.log"), "w").write("\n".join(lines) + "\n")
