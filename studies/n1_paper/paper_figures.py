"""Publication figures for the N1 paper, drawn only from the study's JSON outputs and qwkit.
Writes vector PDFs to ../../paper/figures/."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from qwkit import sp, propagation as pr

here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "..", "..", "paper", "figures"); os.makedirs(outdir, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "lines.linewidth": 1.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
HW = 1.23984 / 10
TW = [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")]


def panel_label(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", color=INK)


def clean(ax, grid_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)


# ---------------- Fig. 1: concept -------------------------------------------------------------
def fig1():
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.12)
    VH = None
    for i, F in enumerate((-55.0, 0.0, 55.0)):
        ax = fig.add_subplot(gs[0, i])
        r = sp.self_consistent(TW, F, dz=0.5)
        E, psi, Z, rho = sp.prepare(r)
        z = r["z"] - 0.5 * (r["active"][0] + r["active"][1])
        V = r["V"] - E[0]
        ax.plot(z / 10, V * 1e3, color=INK2, lw=0.9)
        cols = [BLUE, MUTED, ORANGE, AQUA]
        for k in range(min(4, len(E))):
            dens = psi[:, k] ** 2; dens = dens / dens.max() * 45
            Ek = (E[k] - E[0]) * 1e3
            ax.fill_between(z / 10, Ek, Ek + dens, color=cols[k], alpha=0.35 if k else 0.55, lw=0)
            ax.plot(z / 10, Ek + dens, color=cols[k], lw=0.8)
            ax.text(17.5, Ek + 4, f"{k+1}", color=cols[k], fontsize=7, ha="right", va="bottom")
        ax.set_xlim(-18, 18); ax.set_ylim(-60, 520)
        ax.set_title({-55.0: "$F=-55$ kV/cm", 0.0: "$F=0$", 55.0: "$F=+55$ kV/cm"}[F], color=INK)
        ax.set_xlabel("$z$ (nm)")
        if i == 0:
            ax.set_ylabel("Energy above $E_1$ (meV)")
        else:
            ax.set_yticklabels([])
        clean(ax, grid_y=False)
        if i == 0:
            panel_label(ax, "(a)", x=-0.3)
    # device schematic
    ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    panel_label(ax, "(b)", x=-0.02, y=0.95)
    layers = [(0, 9, "#f1f0ec", "n-InP lower cladding / ground contact"), (9, 13, "#e9e7e1", "InGaAs spacer"),
              (13, 20, "#cfe0f5", "40-period symmetric triple-well stack"), (20, 24, "#e9e7e1", "InGaAs spacer"),
              (24, 31, "#f7f6f2", "InP:Fe semi-insulating upper cladding")]
    for y0, y1, c, lab in layers:
        ax.add_patch(Rectangle((8, y0), 72, y1 - y0, facecolor=c, edgecolor=INK2, lw=0.4))
        ax.text(81, (y0 + y1) / 2, lab, fontsize=6.8, va="center", color=INK2)
    seg = 72 / 6.0
    for j in range(6):
        x0 = 8 + j * seg + 0.6; sgn = 1 if j % 2 == 0 else -1
        ax.add_patch(Rectangle((x0, 31), seg - 1.2, 2.2, facecolor="#c9a227", edgecolor=INK2, lw=0.4))
        ax.text(x0 + (seg - 1.2) / 2, 34.6, "$+V$" if sgn > 0 else "$-V$", ha="center", fontsize=7.5, color=INK)
        a = FancyArrowPatch((x0 + (seg - 1.2) / 2, 16.5 + 2.6 * sgn), (x0 + (seg - 1.2) / 2, 16.5 - 2.6 * sgn),
                            arrowstyle="-|>", mutation_scale=7, color=BLUE, lw=1.0)
        ax.add_patch(a)
        ax.text(x0 + (seg - 1.2) / 2 + 1.4, 17.3, r"$+\chi^{(2)}$" if sgn > 0 else r"$-\chi^{(2)}$", fontsize=6.6, color=BLUE)
    ax.annotate("", xy=(8 + 2 * seg, 37.8), xytext=(8, 37.8), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
    ax.text(8 + seg, 38.6, r"$\Lambda = 2L_{\mathrm{coh}} \approx 47\ \mu$m", ha="center", fontsize=7, color=INK2)
    ax.annotate("", xy=(6.2, 27), xytext=(6.2, 6), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6))
    ax.text(5.0, 16.5, "growth $z$", rotation=90, va="center", ha="right", fontsize=7, color=INK2)
    ax.annotate("", xy=(60, 3.5), xytext=(30, 3.5), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6))
    ax.text(45, 4.6, r"pump $\omega$ (TM) $\rightarrow$ SH $2\omega$", ha="center", fontsize=7, color=INK2)
    fig.savefig(os.path.join(outdir, "fig1_concept.pdf"))
    plt.close(fig)


# ---------------- Fig. 2: chi vs field and mechanism -------------------------------------------
def fig2():
    C = json.load(open(os.path.join(here, "curves_sc.json")))
    fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.7), gridspec_kw=dict(hspace=0.38))
    ax = axs[0]
    styles = {"Symmetric triple well 58/14/34/14/58": (BLUE, 2.0, "Triple well 58/14/34/14/58 Å"),
              "Symmetric double well 56/6/56": (MUTED, 1.2, "Double well 56/6/56 Å"),
              "Single well 116": ("#6d6c67", 1.2, "Single well 116 Å"),
              "Stepped well 50/40/50": ("#9c3d12", 1.2, "Stepped well 50/40/50 Å")}
    for name, rows in C["curves"].items():
        c, lw, lab = styles[name]
        if "Stepped" in name:
            rows = [r for r in rows if r["F"] <= 180]          # upper level unbound beyond 180 kV/cm
        F = [r["F"] for r in rows]; p = [r["pct"] for r in rows]
        ax.plot(F, p, color=c, lw=lw, label=lab, marker="o" if "triple" in name else None, ms=2.6)
    ax.legend(frameon=False, loc="upper right", fontsize=6.2, handlelength=1.6, borderaxespad=0.2)
    for yv, lab in ((30, "Gate (30%)"), (50, "Ties ideal on/off QPM (50%)")):
        ax.axhline(yv, color=INK2, lw=0.6, ls=(0, (4, 3)))
        ax.text(203, yv - 1.2, lab, fontsize=6.5, color=INK2, ha="right", va="top")
    tri = C["curves"]["Symmetric triple well 58/14/34/14/58"]; pk = max(tri, key=lambda r: r["pct"])
    ax.annotate(f"{pk['pct']:.0f}% at {pk['F']} kV/cm", xy=(pk["F"], pk["pct"]), xytext=(4, 67),
                fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    # fields above 140 kV/cm: third level above the barrier edge at the period boundary (outside the single-period model)
    ax.axvspan(140, 205, color="#ecebe7", lw=0, zorder=0)
    ax.text(183.5, 1.2, "third level unbound /\nabove validity limit", fontsize=5.6, color=INK2, ha="center", va="bottom",
            zorder=1, linespacing=1.0)
    ax.set_xlim(0, 205); ax.set_ylim(0, 92)
    ax.set_xlabel("Applied field $F$ (kV/cm)"); ax.set_ylabel(r"$|\chi^{(2)}|$ (% of optimised ADQW)")
    clean(ax); panel_label(ax, "(a)")
    ax = axs[1]
    M = C["mech"]
    F = [m["F"] for m in M]
    for k, (c, lab) in enumerate([(MUTED, "$E_{21}$"), (ORANGE, "$E_{31}$"), (AQUA, "$E_{41}$")], start=1):
        ax.plot(F, [m["E"][k] * 1e3 for m in M], color=c, lw=1.4, label=lab)
    ax.axhline(HW * 1e3, color=INK2, lw=0.6, ls=(0, (4, 3))); ax.text(2, HW * 1e3 + 4, r"$\hbar\omega$", fontsize=7, color=INK2)
    ax.axhline(2 * HW * 1e3, color=INK2, lw=0.6, ls=(0, (4, 3))); ax.text(2, 2 * HW * 1e3 + 4, r"$2\hbar\omega$", fontsize=7, color=INK2)
    ax.set_xlim(0, 110); ax.set_ylim(0, 420)
    ax.set_xlabel("Applied field $F$ (kV/cm)"); ax.set_ylabel("Transition energy (meV)")
    ax2 = ax.twinx()   # same physical quantity family? no: separate axis is a dipole length -> drawn as inset instead
    ax2.remove()
    ins = ax.inset_axes([0.62, 0.72, 0.34, 0.22])
    ins.plot(F, [abs(m["z11"]) / 10 for m in M], color=BLUE, lw=1.2)
    ins.set_xlabel("$F$ (kV/cm)", fontsize=6); ins.set_ylabel(r"$|\langle z\rangle_1|$ (nm)", fontsize=6)
    ins.tick_params(labelsize=5.5); clean(ins, grid_y=False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.45), ncol=3, handlelength=1.4)
    clean(ax); panel_label(ax, "(b)")
    fig.savefig(os.path.join(outdir, "fig2_chi_vs_field.pdf"))
    plt.close(fig)


# ---------------- Fig. 3: error budget -----------------------------------------------------------
def fig3():
    R = json.load(open(os.path.join(here, "..", "n1", "errbudget.json")))
    groups = [("Band offset ±0.02 eV", "band offset"), ("Well mass ±5%", "well mass"), ("Barrier mass ±5%", "barrier mass"),
              ("Linewidth 6–14 meV", "linewidth"), ("TW outer wells ±1 ML", "triple well outer"),
              ("TW barriers ±1 ML", "triple well barriers"), ("TW centre well ±1 ML", "triple well centre"),
              ("ADQW wide well ±1 ML", "ADQW wide"), ("ADQW barrier ±1 ML", "ADQW barrier"), ("ADQW narrow well ±1 ML", "ADQW narrow")]
    nom = R[0]["ratio_same_case"] * 100
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for i, (lab, key) in enumerate(groups):
        v = [r["ratio_same_case"] * 100 for r in R if r["case"].startswith(key)]
        y = len(groups) - 1 - i
        c = BLUE if lab.startswith("TW") else (ORANGE if lab.startswith("ADQW") else MUTED)
        ax.plot([min(v), max(v)], [y, y], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(v, [y] * len(v), "o", color=c, ms=4)
    ax.axvline(30, color=INK2, lw=0.7, ls=(0, (4, 3))); ax.text(30.8, len(groups) - 0.35, "gate", fontsize=6.5, color=INK2)
    ax.axvline(nom, color=INK2, lw=0.7); ax.text(nom + 0.8, len(groups) - 0.35, f"nominal {nom:.0f}%", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[0] for g in groups][::-1], fontsize=7)
    ax.set_xlim(20, 115); ax.set_ylim(-0.6, len(groups) - 0.1)
    ax.set_xlabel(r"Triple-well $|\chi^{(2)}|$ as % of ADQW (same case)")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.savefig(os.path.join(outdir, "fig3_error_budget.pdf"))
    plt.close(fig)


# ---------------- Fig. 4: waveguide and electrodes ---------------------------------------------
def fig4():
    W = [w for w in json.load(open(os.path.join(here, "waveguide.json"))) if w["t_up"] == 2.0 and w["t_sp"] == 0.9][0]
    E = json.load(open(os.path.join(here, "electrodes.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1.25]))
    ax = axs[0]
    z = np.array(W["prof"]["z"]); e1 = np.array(W["prof"]["Ez1"]) ** 2; e2 = np.array(W["prof"]["Ez2"]) ** 2
    zt = z - W["z0"]                                   # depth relative to stack top
    ax.axvspan(0, W["z1"] - W["z0"], color="#cfe0f5", lw=0)
    ax.text((W["z1"] - W["z0"]) / 2, 1.04, "QW stack", ha="center", fontsize=6.8, color=INK2, transform=ax.get_xaxis_transform())
    ax.plot(zt, e1 / e1.max(), color=BLUE, label=r"pump, $\lambda=10\ \mu$m")
    ax.plot(zt, e2 / e2.max(), color=ORANGE, label=r"SH, $\lambda=5\ \mu$m")
    ax.set_xlim(-3.2, 7.0); ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"Depth below stack top ($\mu$m)"); ax.set_ylabel(r"$|E_z|^2$ (normalised)")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.98, 0.55, f"$\\Gamma_\\omega$={W['G1']:.2f}, $\\Gamma_{{2\\omega}}$={W['G2']:.2f}\n"
            f"$L_\\mathrm{{coh}}$={W['Lcoh_um']:.1f} $\\mu$m\nloss {W['a1']:.1f} / {W['a2']:.2f} cm$^{{-1}}$",
            transform=ax.transAxes, ha="right", fontsize=6.6, color=INK2)
    clean(ax, grid_y=False); panel_label(ax, "(a)", x=-0.2)
    ax = axs[1]
    P = E["profile"]; x = np.array(P["x_um"]); F = np.array(P["F"])
    ax.plot(x, -F, color=INK2, lw=1.2, label="field at stack centre")
    ax.set_xlim(0, E["Lam_um"]); ax.set_ylabel("Field $F$ (kV/cm)"); ax.set_xlabel(r"Position along waveguide ($\mu$m)")
    ax.axhline(0, color=GRID, lw=0.6)
    ax2 = ax.twinx()
    ax2.plot(x, -np.array(P["chi"]), color=BLUE, lw=1.6, label=r"$\chi^{(2)}/|\chi^{(2)}|_{\max}$")
    ax2.set_ylim(-1.15, 1.15); ax2.set_ylabel(r"$\chi^{(2)} / |\chi^{(2)}|_{\max}$", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    lim = 1.15 * np.abs(F).max(); ax.set_ylim(-lim, lim)
    g = [r for r in E["results"] if r["gap_um"] == 2.0][0]
    ax.text(0.96, 0.93, f"{g['V']:.0f} V segments, 2 $\\mu$m gaps\nQPM coefficient = {g['rel_ideal']*100:.0f}% of ideal",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK)
    for s in ("top",): ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    panel_label(ax, "(b)", x=-0.16)
    fig.savefig(os.path.join(outdir, "fig4_waveguide_electrodes.pdf"))
    plt.close(fig)


# ---------------- Fig. 5: saturated propagation --------------------------------------------------
def fig5():
    """Electrical schemes from realfield_eff.json (propagated through the electrode fields, 188 kV/cm breakdown limit);
    growth reversal, idealized on/off, phase matching and no QPM (no electrodes) from propagation_lo.json."""
    D = json.load(open(os.path.join(here, "propagation_lo.json"))); R = [r for r in D["results"] if r["design"] == "ad"]
    EF = json.load(open(os.path.join(here, "realfield_eff.json")))
    for e in EF:
        R += [dict(ns=e["ns"], P=e["P"], eta=e["tw55"]["real"]["eta"], design="tw", scheme="sign"),
              dict(ns=e["ns"], P=e["P"], eta=e["twopt"]["eta"], design="tw", scheme="sign_opt"),
              dict(ns=e["ns"], P=e["P"], eta=e["adbip"]["eta"], design="ad", scheme="bipolar"),
              dict(ns=e["ns"], P=e["P"], eta=e["stark"]["eta"], design="ad", scheme="stark")]
    T = json.load(open(os.path.join(here, "tables.json")))
    RATES = json.load(open(os.path.join(here, "rates.json")))
    sch = [(("tw", "sign"), BLUE, "-", "Triple well, sign reversal, 55 kV/cm"),
           (("tw", "sign_opt"), BLUE, (0, (4, 1.5)), "Triple well, sign reversal, bias optimized"),
           (("ad", "bipolar"), "#1baf7a", "-", "ADQW, bipolar $\\pm F$ (bias optimized)"),
           (("ad", "stark"), ORANGE, "-", "ADQW, 0 V / optimized off field"),
           (("ad", "onoff"), ORANGE, (0, (1, 1.4)), "ADQW, on/off QPM, on-state loss throughout"),
           (("ad", "sign"), "#9c3d12", (0, (5, 1.5)), "ADQW, ideal growth-defined orientation reversal"),
           (("ad", "perfect"), MUTED, (0, (3, 2)), "ADQW, perfect phase matching (bound)"),
           (("ad", "none"), MUTED, "-", "ADQW, no QPM")]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw=dict(wspace=0.38))
    ax = axs[0]
    for key, c, ls, lab in sch:
        rr = sorted([r for r in R if (r["design"], r["scheme"]) == key and abs(r["ns"] - 5e11) < 1], key=lambda r: r["P"])
        ax.loglog([r["P"] for r in rr], [r["eta"] * 100 for r in rr], color=c, ls=ls, lw=1.4, marker="o", ms=2.0)
    ax.set_xlabel("Pump power (W)"); ax.set_ylabel("Peak conversion efficiency (%)")
    ax.set_title(r"$n_s = 5\times10^{11}$ cm$^{-2}$ per period", color=INK)
    clean(ax); panel_label(ax, "(a)", x=-0.27)
    ax = axs[1]
    for key, c, ls, lab in sch:
        rr = [r for r in R if (r["design"], r["scheme"]) == key]
        ns = sorted(set(r["ns"] for r in rr))
        Pref = min(set(r["P"] for r in rr), key=lambda p: abs(p - 10.0))
        ax.loglog(ns, [[r["eta"] for r in rr if r["ns"] == n and abs(r["P"] - Pref) < 1e-9][0] * 100 for n in ns],
                  color=c, ls=ls, lw=1.4, marker="o", ms=2.4)
    ax.set_xlabel(r"Sheet density per period (cm$^{-2}$)"); ax.set_ylabel("Peak conversion efficiency (%)")
    ax.set_title(f"pump {Pref:.1f} W", color=INK)
    clean(ax); panel_label(ax, "(b)", x=-0.27)
    ax = axs[2]
    e = [x for x in EF if abs(x["ns"] - 5e11) < 1 and abs(x["P"] - Pref) < 1e-9][0]
    i = [k for k, s_ in enumerate(T["states"]) if abs(s_["ns"] - 5e11) < 1][0]
    st = T["states"][i]
    from propagation_study import wg_for
    m_ad = pr.Medium(st["ad"], HW, R=[r for r in RATES if r["i"] == i and r["design"] == "ad"][0]["R"])
    none = pr.propagate(m_ad, wg_for(st["ad"]), Pref, "none", L_max_um=400.0)
    for x, y, (_, c, ls, _) in ((e["tw55"]["real"]["x_um"], e["tw55"]["real"]["P2"], sch[0]),
                                (e["stark"]["real"]["x_um"], e["stark"]["real"]["P2"], sch[3]),
                                (none["x_um"], none["P2"], sch[7]),
                                (e["adbip"]["real"]["x_um"], e["adbip"]["real"]["P2"], sch[2])):
        ax.plot(x, np.array(y) / Pref * 100, color=c, ls=ls, lw=1.3)
    ax.set_xlabel(r"Propagation distance ($\mu$m)"); ax.set_ylabel("SH power / input pump (%)")
    ax.set_title(f"$n_s = 5\\times10^{{11}}$, pump {Pref:.1f} W", color=INK)
    ax.set_xlim(0, 400); clean(ax); panel_label(ax, "(c)", x=-0.27)
    h = [plt.Line2D([], [], color=c, ls=ls, lw=1.5) for _, c, ls, _ in sch]
    fig.legend(h, [s_[3] for s_ in sch], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.36), fontsize=6.4)
    fig.savefig(os.path.join(outdir, "fig5_propagation.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}[w](); print("fig", w, "done", flush=True)
