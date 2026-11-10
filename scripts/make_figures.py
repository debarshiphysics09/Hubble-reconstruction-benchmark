"""Regenerate every manuscript figure from results/*.csv (run the run_* scripts first)."""
from _common import *
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
TICKS = [0.3, 1, 3, 10, 30, 100, 1000, 10000]


def logx(ax, lo, hi):
    ax.set_xscale("log"); ax.set_xlim(lo, hi)
    t = [x for x in TICKS if lo <= x <= hi]
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in t]))
    ax.xaxis.set_minor_formatter(NullFormatter())
from hzbench.cosmology import FLAT_LCDM
from hzbench.data_gen import generate_Hz_data
from hzbench.metrics import deceleration_parameter, flat_lcdm_model, parametric_fit
from hzbench.real_data import load_cc_data
from hzbench.real_bao_data import load_bao_data

plt.rcParams.update({
    "font.size": 11, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.titlesize": 11, "axes.labelsize": 12, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.7,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
C = {"GP": "#009E73", "Polynomial": "#E69F00", "NN": "#0072B2", "Spline": "#CC79A7",
     "Nodal": "#D55E00", "Nodal-range": "#8C3B00", "ParametricLCDM": "#555555"}
LAB = {"ParametricLCDM": "Parametric", "Nodal": "Nodal (v1, [0,2])", "Nodal-range": "Nodal (data range)"}
os.makedirs("figures", exist_ok=True)
R = lambda f: pd.read_csv(f"results/{f}")
ZG = np.linspace(0, 2, 400)


def save(fig, name):
    fig.savefig(f"figures/{name}"); plt.close(fig); log("saved", name)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(-1.2, 6.3)
    layers = [(1.2, 1, "#009E73"), (4.0, 6, "#0072B2"), (6.8, 6, "#003a5c"), (9.0, 1, "#D55E00")]
    pos = []
    for x, n, c in layers:
        ys = [3.0 + ((n - 1) / 2 - i) * 0.75 for i in range(n)]
        pos.append([(x, y, c) for y in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1, _) in a:
            for (x2, y2, _) in b:
                ax.plot([x1, x2], [y1, y2], color="#d0d0d0", lw=0.5, zorder=1)
    for L in pos:
        for x, y, c in L:
            ax.scatter([x], [y], s=330, color=c, zorder=3, edgecolor="white", lw=1.2)
    ax.text(4.0, 3.0 + 2.5 * 0.75 + 0.6, "...", ha="center"); ax.text(6.8, 3.0 + 2.5 * 0.75 + 0.6, "...", ha="center")
    for x, t in [(1.2, "Input\n$z$"), (4.0, "Hidden layer 1\n64 units, tanh"),
                 (6.8, "Hidden layer 2\n64 units, tanh"), (9.0, "Output\n$\\hat H(z)$, linear")]:
        ax.text(x, -0.15, t, ha="center", va="top", fontsize=10)
    ax.text(5.1, 6.0, r"Adam, $\alpha=10^{-3}$;  $\ell_2$ weight decay $10^{-4}$;  2000 epochs",
            ha="center", fontsize=10)
    ax.text(5.1, 5.45, r"Trainable parameters: $(64+64)+(64^2+64)+(64+1)=4{,}353$", ha="center", fontsize=10)
    save(fig, "fig01_nn_architecture.pdf")


def fig_reconstructions():
    z, H, s, _ = generate_Hz_data(FLAT_LCDM, n=80, sigma=3.0, seed=0)
    t1 = R("tier1_main.csv"); r0 = t1[(t1.noise == 3) & (t1.seed == 0)].set_index("method").rmse
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True); axes = axes.ravel()
    for ax, name in zip(axes, ORDER):
        ax.errorbar(z, H, s, fmt=".", color="#999999", ms=3, lw=0.6, label="Mock data")
        ax.plot(ZG, FLAT_LCDM.Hz(ZG), "k--", lw=1.1, label="Truth")
        ax.plot(ZG, FITTERS[name](z, H, s)(ZG), color=C[name], label=name)
        ax.set_title(f"{name} (RMSE = {r0[name]:.2f})"); ax.legend(loc="upper left")
    for ax in axes[3:5]: ax.set_xlabel("$z$")
    for ax in axes[[0, 3]]: ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    axes[5].axis("off"); fig.tight_layout(); save(fig, "fig02_single_realisation.pdf")


def fig_rmse_vs_noise():
    t1 = R("tier1_main.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for name in ORDER + ["ParametricLCDM"]:
        g = t1[t1.method == name].groupby("noise").rmse
        med, lo, hi = g.median(), g.quantile(0.16), g.quantile(0.84)
        ax.plot(med.index, med, "o-", color=C[name], label=LAB.get(name, name).replace(" (v1, [0,2])", ""))
        ax.fill_between(med.index, lo, hi, color=C[name], alpha=0.12)
    ax.set_xlabel(r"Noise level $\sigma$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"Integrated RMSE [km s$^{-1}$ Mpc$^{-1}$]"); ax.set_yscale("log"); ax.legend(ncol=2)
    save(fig, "fig03_rmse_vs_noise.pdf")


def fig_tier2():
    t2 = R("tier2_robust.csv")
    panels = [("Hz", ["flat_lcdm", "nonflat_lcdm", "cpl_de", "feature"]), ("DH", ["flat_lcdm", "nonflat_lcdm", "cpl_de"])]
    names = {"flat_lcdm": "Flat ΛCDM", "nonflat_lcdm": "Non-flat ΛCDM", "cpl_de": "CPL", "feature": "Local feature"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[4, 3]))
    meths = ORDER + ["Param_correct"]
    for ax, (obs, cos) in zip(axes, panels):
        w = 0.14
        for j, m in enumerate(meths):
            vals = [t2[(t2.observable == obs) & (t2.cosmology == c) & (t2.method == m)].rmse.median() for c in cos]
            col = "#555555" if m == "Param_correct" else C[m]
            ax.bar(np.arange(len(cos)) + (j - 2.5) * w, vals, w, color=col,
                   label="Parametric (correct family)" if m == "Param_correct" else m)
        ax.set_xticks(range(len(cos))); ax.set_xticklabels([names[c] for c in cos])
        ax.set_title(r"Direct $H(z)$, $\sigma=3$" if obs == "Hz" else r"BAO $D_H$ proxy, 3%")
        ax.set_ylabel("Median RMSE [km s$^{-1}$ Mpc$^{-1}$]")
    axes[0].legend(ncol=3, fontsize=7.5); fig.tight_layout(); save(fig, "fig04_tier2.pdf")


def fig_zt():
    t1 = R("tier1_main.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, s in zip(axes, [1, 3, 5]):
        d = [t1[(t1.noise == s) & (t1.method == m)].zt_hat.dropna().values for m in ORDER]
        bp = ax.boxplot(d, patch_artist=True, widths=0.6, medianprops=dict(color="k"),
                        flierprops=dict(marker="o", ms=3))
        for p, m in zip(bp["boxes"], ORDER): p.set_facecolor(C[m]); p.set_alpha(0.6)
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(ORDER, rotation=30)
        ax.axhline(FLAT_LCDM.transition_redshift(), ls="--", color="k", lw=1)
        ax.set_title(rf"$\sigma={s}$")
    axes[0].set_ylabel(r"$\hat z_{\rm t}$"); fig.tight_layout(); save(fig, "fig05_zt.pdf")


def fig_qz():
    z, H, s, _ = generate_Hz_data(FLAT_LCDM, n=80, sigma=3.0, seed=0)
    zq = np.linspace(0.05, 1.95, 400)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(zq, FLAT_LCDM.q(zq), "k--", label="Truth")
    for m in ORDER:
        ax.plot(zq, deceleration_parameter(FITTERS[m](z, H, s), zq), color=C[m], label=m, lw=1.3)
    ax.axhline(0, color="grey", lw=0.6); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("$z$"); ax.set_ylabel("$q(z)$"); ax.legend(ncol=2); save(fig, "fig06_qz.pdf")


def fig_ratio():
    t1 = R("tier1_main.csv")
    p = t1.pivot_table(index=["noise", "seed"], columns="method", values="rmse")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in ORDER:
        rr = (p[m] / p["ParametricLCDM"]).groupby("noise")
        ax.plot(rr.median().index, rr.median(), "o-", color=C[m], label=f"{m}: median of ratios")
        rm = p[m].groupby("noise").mean() / p["ParametricLCDM"].groupby("noise").mean()
        ax.plot(rm.index, rm, "s:", color=C[m], ms=4, label=f"{m}: ratio of means")
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$ [km s$^{-1}$ Mpc$^{-1}$]"); ax.set_ylabel("RMSE / RMSE(parametric)")
    ax.legend(ncol=2, fontsize=7); save(fig, "fig07_cost_ratio.pdf")


def fig_loss():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for s, c in zip([1, 3, 5], ["#009E73", "#E69F00", "#D55E00"]):
        z, H, sg, _ = generate_Hz_data(FLAT_LCDM, n=80, sigma=float(s), seed=0)
        _, L = M.fit_nn(z, H, sg, return_losses=True)
        ax.plot(L, color=c, label=rf"$\sigma={s}$")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted MSE (normalised units)")
    ax.legend(); save(fig, "fig08_training_loss.pdf")


def _data(name):
    if name.startswith("CC"):
        return load_cc_data(name)
    return load_bao_data(name)[:3]


def fig_real_recon(datasets, fname, title_key):
    summ = R("real_loocv_summary.csv")
    zs, Hs, ss, kinds = [], [], [], []
    for d in datasets:
        z, H, s = _data(d); zs.append(z); Hs.append(H); ss.append(s); kinds += [d] * len(z)
    z = np.concatenate(zs); o = np.argsort(z); H = np.concatenate(Hs)[o]; s = np.concatenate(ss)[o]
    kinds = np.array(kinds)[o]; z = z[o]
    zg = np.linspace(0, z.max() + 0.05, 400)
    pf = parametric_fit(z, H, s)[0]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True); axes = axes.ravel()
    meths = ORDER + ["Nodal-range"]
    fit = dict(FITTERS); zr = (z.min(), z.max() + 1e-9)
    fit["Nodal-range"] = lambda a, b, c: M.fit_nodal(a, b, c, zmin=zr[0], zmax=zr[1])
    for ax, m in zip(axes, meths):
        for d, mk, col in zip(datasets, ["o", "s"], ["#888888", "#1f77b4"]):
            k = kinds == d
            ax.errorbar(z[k], H[k], s[k], fmt=mk, ms=3, color=col, lw=0.7, label={"desi": "DESI DR2"}.get(d, d))
        ax.plot(zg, pf(zg), "k--", lw=1, label="Flat ΛCDM fit")
        ax.plot(zg, fit[m](z, H, s)(zg), color=C[m], label=LAB.get(m, m))
        chi = summ[(summ.dataset == title_key) & (summ.method == m)].loocv_chi.values[0]
        ax.set_title(f"{LAB.get(m, m)}  ($\\chi_{{\\rm LOO}}$ = {chi:.2f})", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=7); ax.set_ylim(30, 280)
    for ax in axes[3:]: ax.set_xlabel("$z$")
    for ax in axes[[0, 3]]: ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    fig.tight_layout(); save(fig, fname)


def fig_loocv_bars():
    summ = R("real_loocv_summary.csv")
    sets = ["CC31", "BAO_legacy_v1", "BAO_DESI", "CC31+BAO_DESI"]
    titles = ["CC31 (N=31)", "BAO, v1 table (N=10)", "BAO, DESI DR2 (N=6)", "CC31 + DESI DR2 (N=37)"]
    meths = ["ParametricLCDM"] + ORDER + ["Nodal-range"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, d, t in zip(axes, sets, titles):
        sub = summ[summ.dataset == d].set_index("method")
        y = np.arange(len(meths))
        v = np.array([sub.loc[m, "loocv_chi"] for m in meths])
        lo = v - np.array([sub.loc[m, "ci_lo"] for m in meths]); hi = np.array([sub.loc[m, "ci_hi"] for m in meths]) - v
        ax.barh(y, v, xerr=[lo, hi], color=[C[m] for m in meths], capsize=2)
        ax.set_yticks(y); ax.set_yticklabels([LAB.get(m, m) for m in meths] if ax is axes[0] else [])
        logx(ax, 0.25, max(30, 1.5 * float(np.max(v + hi)))); ax.axvline(1, color="k", lw=0.7, ls=":")
        ax.set_title(t, fontsize=10); ax.set_xlabel(r"$\chi_{\rm LOO}$ (log scale)")
    fig.tight_layout(); save(fig, "fig11_loocv_bars.pdf")


def fig_nodal_diag():
    p = R("nodal_diagnostics_points.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, d, t in zip(axes, ["BAO_legacy_v1", "BAO_legacy", "BAO_desi"],
                        ["BAO v1 table (N=10)", "BAO legacy, corrected (N=7)", "BAO DESI DR2 (N=6)"]):
        for v, mk, col in [("v1 [0,2] K8", "o", "#D55E00"), ("range K8", "s", "#8C3B00"), ("quantile K4", "^", "#7570b3")]:
            q = p[(p.dataset == d) & (p.variant == v)]
            ax.scatter(q.z, q.resid, marker=mk, color=col, s=28, label=v, alpha=0.85,
                       edgecolor=["k" if e else "none" for e in q.extrapolated])
        ax.axhline(0, color="k", lw=0.6); ax.set_yscale("symlog", linthresh=2)
        ax.set_title(t, fontsize=10); ax.set_xlabel("$z$ of held-out point")
    axes[0].set_ylabel(r"LOOCV residual $(\hat H_{-i}-H_i)/\sigma_i$"); axes[0].legend(fontsize=7.5)
    fig.tight_layout(); save(fig, "fig12_nodal_diagnostics.pdf")


def fig_control():
    sc = R("synthetic_control.csv"); summ = R("real_loocv_summary.csv")
    designs = [("CC31", "CC31"), ("BAO_legacy_v1", "BAO_legacy_v1"), ("BAO_desi", "BAO_DESI"), ("CC31+BAO_desi", "CC31+BAO_DESI")]
    meths = ["ParametricLCDM"] + ORDER + ["Nodal-range"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=False)
    for ax, (d, real) in zip(axes, designs):
        sub = sc[sc.design == d]
        data = [np.clip(sub[sub.method == m].loocv_chi.values, 1e-2, 1e3) for m in meths]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55, medianprops=dict(color="k"),
                        flierprops=dict(marker=".", ms=3))
        for b, m in zip(bp["boxes"], meths): b.set_facecolor(C[m]); b.set_alpha(0.45)
        rs = summ[summ.dataset == real].set_index("method")
        ax.scatter([rs.loc[m, "loocv_chi"] for m in meths], range(1, len(meths) + 1), marker="*", s=70,
                   color="k", zorder=5, label="real data")
        allv = np.concatenate(data); logx(ax, 0.25, max(30, 1.5 * float(np.max(allv)))); ax.axvline(1, ls=":", color="k", lw=0.7)
        ax.set_yticks(range(1, len(meths) + 1))
        ax.set_yticklabels([LAB.get(m, m) for m in meths] if ax is axes[0] else [])
        ax.set_title(f"design: {d.replace('_', ' ')}", fontsize=9.5); ax.set_xlabel(r"$\chi_{\rm LOO}$")
    axes[0].legend(loc="lower right", fontsize=7.5); fig.tight_layout(); save(fig, "fig13_synthetic_control.pdf")


def fig_calibration():
    f = R("real_loocv_folds.csv"); r = R("synthetic_control_resid.csv")
    from scipy.stats import norm
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    xx = np.linspace(-4, 4, 200)
    for ax, (real, des, t) in zip(axes, [("CC31", "CC31", "CC31"), ("BAO_legacy_v1", "BAO_legacy_v1", "BAO v1 table"),
                                        ("BAO_DESI", "BAO_desi", "BAO DESI DR2")]):
        a = f[(f.dataset == real) & (f.method == "GP")].resid.values
        b = r[(r.design == des) & (r.method == "GP")].resid.values
        ax.hist(np.clip(b, -4, 4), bins=24, range=(-4, 4), density=True, color="#bbbbbb", label="synthetic control (GP)")
        ax.hist(np.clip(a, -4, 4), bins=16, range=(-4, 4), density=True, histtype="step", color="#009E73", lw=2, label="real data (GP)")
        ax.plot(xx, norm.pdf(xx), "k--", lw=1, label="N(0,1)"); ax.set_title(t); ax.set_xlabel("standardised LOOCV residual")
    axes[0].set_ylabel("density"); axes[0].legend(fontsize=7.5); fig.tight_layout(); save(fig, "fig14_calibration.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    jobs = dict(arch=fig_architecture, recon=fig_reconstructions, noise=fig_rmse_vs_noise, tier2=fig_tier2,
                zt=fig_zt, qz=fig_qz, ratio=fig_ratio, loss=fig_loss,
                cc=lambda: fig_real_recon(["CC31"], "fig09_real_cc.pdf", "CC31"),
                ccdesi=lambda: fig_real_recon(["CC31", "desi"], "fig10_real_cc_desi.pdf", "CC31+BAO_DESI"),
                bars=fig_loocv_bars, nodal=fig_nodal_diag, control=fig_control, calib=fig_calibration)
    for k, f in jobs.items():
        if "all" in which or k in which:
            try:
                f()
            except FileNotFoundError as e:
                log(f"skip {k}: {e}")
