"""N1 supporting numbers that no other study script prints (number audit of paper/main.tex).
Everything is computed from the existing study outputs (tables.json, rates.json, propagation_lo.json,
bipolar_fair.json, stark_valid.json, twofield_qpm.json, electrodes.json, ../nl02/qwkit_validation.json and
../nl02/numbers.json) or directly with qwkit. Sections (keys of extras_audit.json):
  bipolar_phase   Sec. 'Why the bipolar ADQW wins': for the field amplitude F that is optimal for the bipolar ADQW
                  at each doping and pump power above 1 W (bipolar_fair.json), the small-signal phase difference
                  of chi2(+F) and chi2(-F) at the resonant pump, each segment's |chi2| relative to the 0 V value
                  at the same doping, and the detunings E~21 - hw and E~31 - 2hw of both segments (meV).
  opt_detuning    Sec. 'Mechanism': E~21 - hw and E~31 - 2hw of the segments of the best two-field ADQW patterns.
  rate_limit      Sec. 'Detuned pumping': conservative rate-equation limit P_valid (Rabi energy = detuned linewidth)
                  with the NL-02 definition (collect_numbers.py): P_valid = pcoh min(1 + d12^2, (1 + d23^2)/r23) Ps,
                  d in units of gamma = 10 meV at the detuned pump, ladder maps of NL-02 (qwkit_validation.mapped_ladder,
                  resonant waveguide as in NL-02). Evaluated for each entry of Table 'detuned' (TW at 55 kV/cm,
                  ADQW at 0 V for growth reversal and phase matching, both Stark segments, limit = smaller one).
  relaxation      Sec. 'Relaxation and saturation': TW partial lifetimes, equilibrium population of subband 2,
                  pump cross-section ratio, pump intensity above which |chi2_TW| > |chi2_ADQW| (LO rates, pump only),
                  stack intensity per watt.
  crossover       Pump power/intensity above which the bias-optimized TW beats growth reversal.
  electrical      Bias voltages for the bias-optimized TW fields of Table 'eff' (electrode plateau field of
                  electrodes.json), and with contact layers adjacent to the stack.
  leakage         Thermionic hopping time over the downhill barrier (attempt rate 1e13 /s, as in n1/post.py) at 55 kV/cm
                  (self-consistent), leakage current, temperature at which the hopping time is 1 ms, and WKB
                  tunnelling broadening of the lowest four localized subbands.
  bulk_lcoh       Coherence length of bulk InP (Pettit-Turner) for the 10 um pump.
  kane            Two-band Kane vs one-band Loewdin dipoles for the ADQW 64/10/29 (method of studies/kane).
  lo_benchmark    10 nm GaAs well LO-phonon benchmark (tests/test_scattering.py).
  bipolar_best    Whether the bipolar ADQW is the most efficient scheme at every doping above 1 W.
  json_checks     Ranges read directly from existing outputs (coherence lengths, Stark off fields, two-field shortfall).
  software        qwkit version and number of test functions.
Writes extras_audit.json. Run from this folder with PYTHONPATH set to the repository root."""
import sys, os, json, glob, re
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(here, "..", "..")
sys.path.insert(0, root); sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, "..", "nl02"))
import numpy as np
from multiprocessing import Pool
import qwkit
from qwkit import sp, core as q, scattering as sc, propagation as pr, waveguide as wgm
import propagation_study as ps
import twofield_qpm as tq
from qwkit_validation import mapped_ladder, ladder_idx

GAM = 0.010; HW = ps.HW; KT = 8.617333e-5
T = ps.T["states"]
J = lambda f: json.load(open(os.path.join(here, f)))
NL = json.load(open(os.path.join(here, "..", "nl02", "numbers.json")))
RATES = J("rates.json")


def chi(st, hw=HW):
    return complex(sp.chi2_depol(np.array(st["Et"]), np.array(st["Z"]), np.array(st["rho"]), hw, GAM, st["N"]))


def detunings(st):
    """E~21 - hw and E~31 - 2hw (meV) for the pump ladder (0, b, c) chosen at the resonant pump."""
    a, b, c = ladder_idx(st, HW); Et = np.array(st["Et"])
    return dict(ladder=[a, b, c], d21_meV=float((Et[b, a] - HW) * 1e3), d31_meV=float((Et[c, a] - 2 * HW) * 1e3))


def p_valid(mp, st, dhw):
    """NL-02 rate-equation limit (W) for ladder map mp (pars, Ps_W, tau21_ps) of state st at pump HW + dhw."""
    a, b, c = ladder_idx(st, HW); Et = np.array(st["Et"]); hw = HW + dhw
    d12 = (Et[b, a] - hw) / GAM; d13 = (Et[c, a] - 2 * hw) / GAM; d23 = d13 - d12
    pcoh = GAM * 1e3 * mp["tau21_ps"] / 0.6582119569          # gamma tau21 / hbar (meV ps / meV ps)
    r23 = mp["pars"]["r23"]
    return float(pcoh * min(1 + d12 ** 2, (1 + d23 ** 2) / r23) * mp["Ps_W"]), float(d12), float(d23)


def sc_state(args):
    return tq.state(args)


def tw55_raw(_):
    lay = [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")]
    r = sp.self_consistent(lay, 55.0, 5e11, maxit=400, nstates=10)
    z, act, V, E = r["z"], r["active"], r["V"], r["E"]
    keep = np.where(q.localized(z, r["psi"], act))[0]
    edge = np.argmax(z > act[1])                      # first spacer cell on the downhill side (F > 0)
    U = float(V[edge] - E[keep[0]])
    esc = []
    for k in keep[:4]:
        G, Uk, d = q.escape_broadening(r, int(k), 55.0)
        esc.append(dict(level=int(k), Gamma_meV=float(G), U_eV=float(Uk)))
    return dict(U_eV=U, escape=esc)


def kane(spacer):
    lay = [(64, "well"), (10, "barrier"), (29, "well")]
    r = sp.self_consistent(lay, 0.0, ns_cm2=5e11, T=300.0, spacer=spacer, dz=0.5, doping="spacer", nstates=10)
    z, psi, E, V, m, Eg, act = r["z"], r["psi"], r["E"], r["V"], r["m"], r["Eg"], r["active"]
    dz = z[1] - z[0]; zc = 0.5 * (act[0] + act[1])
    psi = psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz)
    keep = q.localized(z, psi, act)
    Ek, Pk = E[keep][:4], psi[:, keep][:, :4]
    ZK, ov = q.kane_dipoles(z, Pk, Ek, V, m, Eg, zc)
    E4, P4, ZL, rho = sp.prepare(r, nkeep=4)
    hw = np.arange(0.5, 1.5, 0.002) * (E4[1] - E4[0])
    cL = max(abs(q.chi2_shg(E4, ZL, h, 0.010, 1e18)) for h in hw)
    cK = max(abs(q.chi2_shg(Ek, ZK, h, 0.010, 1e18)) for h in hw)
    return dict(spacer_A=spacer, two_band_overlap_max=float(ov), peak_chi2_ratio_kane_over_loewdin=float(cK / cL))


def kane_pump(design):
    """|chi2| at the pump, full N1 model (self-consistent, 200 A spacer, Fermi-Dirac populations, depolarization-shifted
    energies of the Loewdin states), with Loewdin dipoles vs two-band Kane dipoles of the same localized states."""
    lay, F = ([(64, "well"), (10, "barrier"), (29, "well")], 0.0) if design == "ad" else \
             ([(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")], 55.0)
    r = sp.self_consistent(lay, F, ns_cm2=5e11, maxit=400, nstates=10)
    z, psi, E, V, m, Eg, act = r["z"], r["psi"], r["E"], r["V"], r["m"], r["Eg"], r["active"]
    dz = z[1] - z[0]; zc = 0.5 * (act[0] + act[1])
    psi = psi / np.sqrt(np.sum(psi ** 2, axis=0) * dz)
    keep = q.localized(z, psi, act); n = min(int(keep.sum()), 6)
    ZK, ov = q.kane_dipoles(z, psi[:, keep][:, :n], E[keep][:n], V, m, Eg, zc)
    E6, P6, ZL, rho = sp.prepare(r, nkeep=n)
    Lp = (act[1] - act[0] + 200.0) * 1e-8; N = 5e11 / Lp
    Et = sp.depol_energies(z, P6, E6, rho, 5e11, eps_r=13.5)
    cL = abs(sp.chi2_depol(Et, ZL, rho, HW, GAM, N)); cK = abs(sp.chi2_depol(Et, ZK, rho, HW, GAM, N))
    return dict(design=design, F=F, n_localized=n, two_band_overlap_max=float(ov), chi_loewdin=float(cL), chi_kane=float(cK),
                ratio_kane_over_loewdin=float(cK / cL))


def job(a):
    kind, arg = a
    if kind == "kane_pump":
        return ("kane_pump", arg, kane_pump(arg))
    if kind == "state":
        return ("state", arg, sc_state(arg))
    if kind == "tw55":
        return ("tw55", None, tw55_raw(None))
    return ("kane", arg, kane(arg))


if __name__ == "__main__":
    out = {}
    BF = J("bipolar_fair.json")
    best_hi = [b for b in BF["best"] if b["P"] > 1.0]
    opt_pairs = sorted(set((next(i for i, s in enumerate(T) if abs(s["ns"] - b["ns"]) < 1), b["F_ad"]) for b in best_hi))
    TQ = J("twofield_qpm.json")["results"]
    two_best = []
    for I in tq.NSI:
        for P in tq.PS:
            b = max([r for r in TQ if r["I"] == I and r["P"] == P and r["kind"] == "ad"], key=lambda r: r["eta"])
            two_best.append(dict(I=I, P=P, F1=b["F1"], F2=b["F2"], dhw=b["dhw"]))
    # Table 'detuned' optima (each scheme at its own best detuning) and the Stark off fields there (stark_offscan)
    SOJ = J("stark_offscan.json")
    okF = {(o["I"], o["F"]) for o in SOJ["offs"] if o["valid"]}
    SO = [r for r in SOJ["results"] if (r["I"], r["F"]) in okF]          # valid off fields only
    DQ = {2: J("detuned_qpm_ns2.json"), 4: J("detuned_qpm.json")}
    stark_opt = []
    for I in (2, 4):
        for P in (1.26, 4.0, 10.0):
            b = max([r for r in SO if r["I"] == I and r["P"] == P], key=lambda r: r["eta"])
            stark_opt.append(dict(I=I, P=P, F=b["F"], dhw=b["dhw"], eta=b["eta"]))
    need = set()
    for I, F in opt_pairs:
        need |= {("ad", I, F), ("ad", I, -F)}
    for t in two_best:
        need |= {("ad", t["I"], t["F1"]), ("ad", t["I"], t["F2"])}
    for s in stark_opt:
        need.add(("ad", s["I"], s["F"]))
    need = sorted(x for x in need if x[2] != 0)
    jobs = [("tw55", None), ("kane", 200.0), ("kane", 100.0), ("kane_pump", "ad"), ("kane_pump", "tw")] + [("state", x) for x in need]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    S = {r[1][1:]: r[2] for r in res if r[0] == "state"}
    tw55 = [r[2] for r in res if r[0] == "tw55"][0]
    kan = [r[2] for r in res if r[0] == "kane"]
    kanp = [r[2] for r in res if r[0] == "kane_pump"]

    # ---- bipolar phase reversal at the optimum fields
    rows = []
    for I, F in opt_pairs:
        a, b = S[(I, F)], S[(I, -F)]
        c0 = abs(chi(T[I]["ad"])); cp, cm = chi(a["st"]), chi(b["st"])
        dphi = float((np.degrees(np.angle(cp)) - np.degrees(np.angle(cm))) % 360)
        rows.append(dict(ns=T[I]["ns"], F=F, valid=bool(a["valid"] and b["valid"]), dphi_deg=dphi,
                         rel_plus=abs(cp) / c0, rel_minus=abs(cm) / c0, half_diff_rel=abs(cp - cm) / 2 / c0,
                         det_plus=detunings(a["st"]), det_minus=detunings(b["st"]), det_0V=detunings(T[I]["ad"])))
    rel = [x for r in rows for x in (r["rel_plus"], r["rel_minus"])]
    d21 = [abs(x) for r in rows for x in (r["det_plus"]["d21_meV"], r["det_minus"]["d21_meV"])]
    d31 = [abs(x) for r in rows for x in (r["det_plus"]["d31_meV"], r["det_minus"]["d31_meV"])]
    out["bipolar_phase"] = dict(rows=rows, F_range=[min(F for _, F in opt_pairs), max(F for _, F in opt_pairs)],
                                dphi_range=[min(r["dphi_deg"] for r in rows), max(r["dphi_deg"] for r in rows)],
                                rel_range=[min(rel), max(rel)],
                                half_diff_rel_range=[min(r["half_diff_rel"] for r in rows), max(r["half_diff_rel"] for r in rows)],
                                abs_d21_range_meV=[min(d21), max(d21)], abs_d31_range_meV=[min(d31), max(d31)],
                                note="all pairs valid" if all(r["valid"] for r in rows) else "contains invalid pairs")
    print(f"bipolar optimum fields {opt_pairs}: phase difference {out['bipolar_phase']['dphi_range'][0]:.1f}-"
          f"{out['bipolar_phase']['dphi_range'][1]:.1f} deg, |chi|/|chi(0 V)| {min(rel):.3f}-{max(rel):.3f}, "
          f"|E21-hw| {min(d21):.1f}-{max(d21):.1f} meV, |E31-2hw| {min(d31):.1f}-{max(d31):.1f} meV", flush=True)
    for r in rows:
        print(f"  ns {r['ns']:.0e} +-{r['F']}: valid {r['valid']} dphi {r['dphi_deg']:.1f} |chi+| {r['rel_plus']:.3f} |chi-| {r['rel_minus']:.3f}"
              f" d21 {r['det_plus']['d21_meV']:+.1f}/{r['det_minus']['d21_meV']:+.1f} d31 {r['det_plus']['d31_meV']:+.1f}/{r['det_minus']['d31_meV']:+.1f} meV", flush=True)

    # ---- best two-field ADQW patterns
    od = []
    for t in two_best:
        s1, s2 = S[(t["I"], t["F1"])], S[(t["I"], t["F2"])]
        od.append(dict(t, det1=detunings(s1["st"]), det2=detunings(s2["st"])))
    d21o = [abs(x) for o in od for x in (o["det1"]["d21_meV"], o["det2"]["d21_meV"])]
    out["opt_detuning"] = dict(rows=od, abs_d21_range_meV=[min(d21o), max(d21o)],
                               abs_d31_range_meV=[min(abs(x) for o in od for x in (o["det1"]["d31_meV"], o["det2"]["d31_meV"])),
                                                  max(abs(x) for o in od for x in (o["det1"]["d31_meV"], o["det2"]["d31_meV"]))])
    for o in od:
        print(f"two-field best ns_i {o['I']} P {o['P']}: ({o['F1']:+d},{o['F2']:+d}) dhw {o['dhw']*1e3:+.0f}: d21 {o['det1']['d21_meV']:+.1f}/"
              f"{o['det2']['d21_meV']:+.1f} d31 {o['det1']['d31_meV']:+.1f}/{o['det2']['d31_meV']:+.1f} meV", flush=True)

    # ---- rate-equation limit at the Table 'detuned' optima
    def scheme_best(I, P, key):
        rr = DQ[I]["results"] if isinstance(DQ[I], dict) and "results" in DQ[I] else DQ[I]
        c = [r for r in rr if r.get("scheme") == key and abs(r["P"] - P) < 1e-9]
        return max(c, key=lambda r: r["eta"])
    lim = []
    for I in (2, 4):
        nk = f"{T[I]['ns']:.0e}"
        for P in (1.26, 4.0, 10.0):
            for key, design in (("tw_sign", "tw"), ("ad_rev", "ad"), ("ad_pm", "ad")):
                b = scheme_best(I, P, key)
                pv, d12, d23 = p_valid(NL[f"{design}_{nk}"], T[I][design], b["dhw"])
                lim.append(dict(ns=T[I]["ns"], P=P, scheme=key, dhw_meV=b["dhw"] * 1e3, eta=b["eta"], P_valid_W=pv, d12=d12, d23=d23))
            s = [x for x in stark_opt if x["I"] == I and x["P"] == P][0]
            pv_on = p_valid(NL[f"ad_{nk}"], T[I]["ad"], s["dhw"])[0]
            off = S[(I, s["F"])]
            mp = mapped_ladder(off["st"], np.array(off["R"]), ps.wg_for(off["st"]))
            pv_off = p_valid(mp, off["st"], s["dhw"])[0]
            lim.append(dict(ns=T[I]["ns"], P=P, scheme="ad_stark", F_off=s["F"], dhw_meV=s["dhw"] * 1e3, eta=s["eta"],
                            P_valid_on_W=pv_on, P_valid_off_W=pv_off, P_valid_W=min(pv_on, pv_off)))
    res0 = {k: NL[k]["P_valid_W"] for k in ("ad_1e+11", "tw_1e+11", "ad_5e+11", "tw_5e+11")}
    big = [x["P_valid_W"] for x in lim if abs(x["dhw_meV"]) >= 19.9]
    m10 = [x["P_valid_W"] for x in lim if abs(abs(x["dhw_meV"]) - 10) < 0.1]
    z0 = [x["P_valid_W"] for x in lim if abs(x["dhw_meV"]) < 0.1]
    out["rate_limit"] = dict(rows=lim, resonant_P_valid_W=res0, range_resonant=[min(res0.values()), max(res0.values())],
                             n_optima=len(lim), n_ge20meV=len(big), range_ge20meV=[min(big), max(big)],
                             range_10meV=[min(m10), max(m10)] if m10 else None, range_0meV=[min(z0), max(z0)] if z0 else None)
    print(f"rate-equation limit: resonant {min(res0.values()):.2f}-{max(res0.values()):.2f} W; optima with |D| >= 20 meV "
          f"({len(big)} of {len(lim)}): {min(big):.1f}-{max(big):.1f} W; at -10 meV: {min(m10):.1f}-{max(m10):.1f} W", flush=True)
    for x in lim:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}, flush=True)

    # ---- relaxation and saturation
    rel_out = {}
    I4 = 4
    R = np.array([r for r in RATES if r["i"] == I4 and r["design"] == "tw"][0]["R"])
    rel_out["tw_3to2_ps"] = float(1e12 / R[2, 1]); rel_out["tw_3to1_ps"] = float(1e12 / R[2, 0])
    rel_out["tw_3_branch_to_2"] = float(R[2, 1] / R[2, :2].sum())
    rel_out["tw_2to1_ps"] = float(1e12 / R[1, 0])
    rel_out["tw_rho2_eq"] = float(T[I4]["tw"]["rho"][1]); rel_out["tw_E21_meV"] = float((T[I4]["tw"]["E"][1] - T[I4]["tw"]["E"][0]) * 1e3)
    Ra = np.array([r for r in RATES if r["i"] == I4 and r["design"] == "ad"][0]["R"])
    rel_out["ad_2to1_ps"] = float(1e12 / Ra[1, 0]); rel_out["ad_3_out_ps"] = float(1e12 / Ra[2, :2].sum())
    mt = pr.Medium(T[I4]["tw"], HW, R=R); ma = pr.Medium(T[I4]["ad"], HW, R=Ra)
    at, bt, ct = ladder_idx(T[I4]["tw"], HW); aa, ba, ca = ladder_idx(T[I4]["ad"], HW)
    rel_out["sigma_pump_ratio_ad_over_tw"] = float(ma.sig1[aa, ba] / mt.sig1[at, bt])
    Ig = np.logspace(9, 12, 601)                                      # W/m^2
    cross = {}
    for I, s in enumerate(T):
        Rt = np.array([r for r in RATES if r["i"] == I and r["design"] == "tw"][0]["R"])
        Rd = np.array([r for r in RATES if r["i"] == I and r["design"] == "ad"][0]["R"])
        m1 = pr.Medium(s["tw"], HW, R=Rt); m2 = pr.Medium(s["ad"], HW, R=Rd)
        f = np.array([abs(m1.chi(m1.populations(x, 0.0))) / abs(m2.chi(m2.populations(x, 0.0))) for x in Ig])
        k = int(np.argmax(f >= 1.0))
        cross[f"{s['ns']:.0e}"] = float(np.exp(np.interp(0.0, np.log(f[[k - 1, k]]), np.log(Ig[[k - 1, k]]))) / 1e10) if k > 0 else None
    rel_out["chi_crossover_MWcm2"] = cross
    W = ps.WG
    rel_out["I_per_W_MWcm2"] = float(W["G1"] / (W["t_stack_um"] * 1e-4 * W["w_um"] * 1e-4) / 1e6)
    out["relaxation"] = rel_out
    print("relaxation:", json.dumps(rel_out, default=lambda v: round(v, 3)), flush=True)

    # ---- crossover of the bias-optimized TW and growth reversal
    PL = J("propagation_lo.json")["results"]
    def pl(ns, d, sch, P):
        return [r for r in PL if abs(r["ns"] - ns) < 1 and r["design"] == d and r["scheme"] == sch and abs(r["P"] - P) < 1e-9][0]["eta"]
    co = {}
    for s in T:
        bb = sorted([b for b in BF["best"] if abs(b["ns"] - s["ns"]) < 1], key=lambda b: b["P"])
        Ps_ = np.array([b["P"] for b in bb]); f = np.array([b["tw"] / pl(s["ns"], "ad", "sign", b["P"]) for b in bb])
        k = int(np.argmax(f >= 1.0))
        Pc = float(np.exp(np.interp(0.0, np.log(f[[k - 1, k]]), np.log(Ps_[[k - 1, k]])))) if k > 0 else None
        co[f"{s['ns']:.0e}"] = dict(P_W=Pc, I_MWcm2=None if Pc is None else Pc * rel_out["I_per_W_MWcm2"])
    co55 = {}
    for s in T:
        Ps_ = np.array(ps.PS); f = np.array([pl(s["ns"], "tw", "sign", P) / pl(s["ns"], "ad", "sign", P) for P in ps.PS])
        k = int(np.argmax(f >= 1.0))
        Pc = float(np.exp(np.interp(0.0, np.log(f[[k - 1, k]]), np.log(Ps_[[k - 1, k]])))) if k > 0 else None
        co55[f"{s['ns']:.0e}"] = dict(P_W=Pc, I_MWcm2=None if Pc is None else Pc * rel_out["I_per_W_MWcm2"])
    out["crossover"] = dict(twopt_vs_rev=co, tw55_vs_rev=co55)
    print("crossover TW opt vs rev (W, MW/cm2):", {k: (round(v["P_W"], 3), round(v["I_MWcm2"], 2)) for k, v in co.items()}, flush=True)
    print("crossover TW 55 vs rev (W, MW/cm2):", {k: (round(v["P_W"], 3), round(v["I_MWcm2"], 2)) for k, v in co55.items()}, flush=True)

    # ---- whether the bipolar ADQW is the most efficient scheme above 1 W
    worst = []
    for b in best_hi:
        others = dict(tw_opt=b["tw"], stark=b["stark"], tw55=pl(b["ns"], "tw", "sign", b["P"]),
                      rev=pl(b["ns"], "ad", "sign", b["P"]), pm=pl(b["ns"], "ad", "perfect", b["P"]), onoff=pl(b["ns"], "ad", "onoff", b["P"]))
        k = max(others, key=others.get)
        worst.append(dict(ns=b["ns"], P=b["P"], runner_up=k, ratio=b["adbip"] / others[k]))
    out["bipolar_best"] = dict(min_ratio_to_runner_up=min(w["ratio"] for w in worst), rows=worst)
    print(f"bipolar ADQW / next-best scheme above 1 W: min {min(w['ratio'] for w in worst):.2f}", flush=True)

    # ---- bias voltages
    EL = [r for r in J("electrodes.json")["results"] if r["gap_um"] == 2.0][0]
    V_per = EL["V"] / EL["Fplateau"]
    Ft = sorted(set(b["F_tw"] for b in BF["best"] if b["P"] in (ps.PS[7], ps.PS[10]) and any(abs(b["ns"] - x) < 1 for x in (1e10, 1e11, 5e11))))
    out["electrical"] = dict(V_per_kVcm=V_per, F_twopt_table=Ft, V_twopt=[min(Ft) * V_per, max(Ft) * V_per],
                             V_design_55=55 * V_per, V_contacts_55=55e3 * W["t_stack_um"] * 1e-4)
    print(f"voltages: TW opt fields {Ft} -> {min(Ft)*V_per:.1f}-{max(Ft)*V_per:.1f} V; 55 kV/cm {55*V_per:.1f} V; "
          f"contacts at the stack {out['electrical']['V_contacts_55']:.1f} V", flush=True)

    # ---- thermionic leakage and tunnelling at 55 kV/cm
    U = tw55["U_eV"]; nu = 1e13
    tau = lambda TK: np.exp(U / (KT * TK)) / nu
    T1ms = U / (KT * np.log(1e-3 * nu))
    out["leakage"] = dict(U_eV=U, tau_hop_300K_s=float(tau(300.0)), J_300K_Acm2=float(1.602176634e-19 * 5e11 / tau(300.0)),
                          tau_hop_200K_s=float(tau(200.0)), T_for_1ms_K=float(T1ms), tunnelling=tw55["escape"])
    print(f"leakage: U {U:.3f} eV, hop time {tau(300.0)*1e6:.2f} us at 300 K (J {out['leakage']['J_300K_Acm2']:.3f} A/cm2), "
          f"{tau(200.0)*1e3:.2f} ms at 200 K, 1 ms at {T1ms:.0f} K; tunnelling Gamma (meV) "
          f"{[round(e['Gamma_meV'], 6) for e in tw55['escape']]}", flush=True)

    # ---- bulk InP coherence length
    dn = wgm.n_inp(5.0) - wgm.n_inp(10.0)
    out["bulk_lcoh"] = dict(dn=float(dn), Lcoh_um=float(10.0 / (4 * dn)))
    print(f"bulk InP Lcoh {10.0/(4*dn):.1f} um", flush=True)

    # ---- Kane check
    out["kane"] = dict(peak=kan, pump=kanp)
    for k in kanp:
        print(f"Kane vs Loewdin, |chi2| at the pump, full model, {k['design']} at {k['F']:.0f} kV/cm: ratio {k['ratio_kane_over_loewdin']:.4f}"
              f" ({k['n_localized']} localized states, two-band overlap {k['two_band_overlap_max']:.1e})", flush=True)
    for k in kan:
        print(f"Kane ADQW 64/10/29 (spacer {k['spacer_A']:.0f} A): peak chi2 ratio {k['peak_chi2_ratio_kane_over_loewdin']:.4f}, "
              f"two-band overlap {k['two_band_overlap_max']:.1e}", flush=True)

    # ---- LO-phonon benchmark (same inputs as tests/test_scattering.py::test_gaas_well_benchmark)
    z = (np.arange(1400) + 0.5) * 0.5
    V = np.where((z > 300) & (z < 400), 0.0, 0.2138)
    E, psi = q.solve_states(z, V, np.full_like(z, 0.07), np.full_like(z, 1.42), nstates=3, nonparabolic=False)
    t21 = 1 / sc.lo_rate(z, psi[:, 1], psi[:, 0], E[1] - E[0], T=0.0, hwlo=0.036, eps_inf=10.89, eps_s=12.5, m_eff=0.07)
    out["lo_benchmark"] = dict(E21_meV=float((E[1] - E[0]) * 1e3), dE21_meV=float((E[1] - E[0]) * 1e3 - 85.9),
                               tau21_ps=float(t21 * 1e12), ref_tau21_ps=0.834, rel_dev=float(abs(t21 * 1e12 / 0.834 - 1)))
    print("LO benchmark:", json.dumps(out["lo_benchmark"], default=lambda v: round(v, 3)), flush=True)

    # ---- ranges read from existing outputs
    SV = J("stark_valid.json")
    lc = [(max(r["Lcoh_on"], r["Lcoh_off"]) / min(r["Lcoh_on"], r["Lcoh_off"])) for r in SV["results"]]
    sh = []
    for I in tq.NSI:
        for P in tq.PS:
            rr = [r for r in TQ if r["I"] == I and r["P"] == P and r["kind"] == "ad"]
            b = max(rr, key=lambda r: r["eta"]); s80 = [r for r in rr if {r["F1"], r["F2"]} == {80, -80} and r["dhw"] == 0][0]
            sh.append(1 - s80["eta"] / b["eta"])
    lcoh = sorted(set(round(r["Lcoh_um"], 2) for r in PL))
    out["json_checks"] = dict(stark_Lcoh_max_ratio=max(lc), stark_off_fields_5e11=sorted(set(r["F_off"] for r in SV["results"] if abs(r["ns"] - 5e11) < 1)),
                              pm80_shortfall_range=[min(sh), max(sh)], Lcoh_media_range_um=[min(lcoh), max(lcoh)])
    print("json checks:", json.dumps(out["json_checks"], default=lambda v: round(v, 3)), flush=True)

    # ---- software
    ntests = sum(len(re.findall(r"^def test_", open(f).read(), flags=re.M)) for f in glob.glob(os.path.join(root, "tests", "test_*.py")))
    out["software"] = dict(version=qwkit.__version__, n_test_functions=ntests)
    print(f"qwkit {qwkit.__version__}, {ntests} test functions", flush=True)
    json.dump(out, open(os.path.join(here, "extras_audit.json"), "w"), indent=1, default=float)
