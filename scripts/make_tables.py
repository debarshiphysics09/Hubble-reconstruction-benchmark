"""Write LaTeX table bodies (results/tables/*.tex) and a numbers summary
(results/key_numbers.json) from results/*.csv."""
from _common import *
import json
import pandas as pd
from hzbench.bootstrap_stats import paired_realisation_test
from hzbench.real_bao_data import load_bao_data, desi_dh_over_rd
from hzbench.real_data import load_cc_data

os.makedirs("results/tables", exist_ok=True)
R = lambda f: pd.read_csv(f"results/{f}")
K = {}
LAB = {"ParametricLCDM": "Parametric (flat $\\Lambda$CDM)", "Nodal": "Nodal (v1: $[0,2]$, $K=8$)",
       "Nodal-range": "Nodal (data range, $K=8$)"}


def w(name, lines):
    open(f"results/tables/{name}.tex", "w").write("\n".join(lines) + "\n"); log("table", name)


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def tier1():
    t = R("tier1_main.csv"); nl = R("tier1_noiseless.csv")
    L = []
    for m in ORDER + ["ParametricLCDM"]:
        cells = []
        for s in [1, 3, 5]:
            x = t[(t.method == m) & (t.noise == s)].rmse
            cells.append(f"${x.mean():.3f}\\pm{x.std():.3f}$ & ${x.median():.3f}$")
        L.append(f"{LAB.get(m, m).replace(' (v1: $[0,2]$, $K=8$)', '')} & " + " & ".join(cells) + r" \\")
    w("tier1_rmse", L)
    L = []
    p = t.pivot_table(index=["noise", "seed"], columns="method", values="rmse")
    for m in ORDER:
        cells = []
        for s in [1, 3, 5]:
            b = nl[(nl.method == m) & (nl.noise == s)].rmse.mean()
            r_mean_of = (p.loc[s, m] / p.loc[s, "ParametricLCDM"]).mean()
            r_med_of = (p.loc[s, m] / p.loc[s, "ParametricLCDM"]).median()
            r_of_mean = p.loc[s, m].mean() / p.loc[s, "ParametricLCDM"].mean()
            cells.append(f"{b:.3f} & {r_mean_of:.1f} & {r_med_of:.1f} & {r_of_mean:.1f}")
            K[f"bias_{m}_{s}"] = b; K[f"ratio_meanof_{m}_{s}"] = r_mean_of; K[f"ratio_ofmean_{m}_{s}"] = r_of_mean
            K[f"ratio_medof_{m}_{s}"] = r_med_of
        L.append(f"{m} & " + " & ".join(cells) + r" \\")
    w("tier1_bias_ratio", L)
    # failures
    for m in ORDER:
        for s in [1, 3, 5]:
            K[f"fail_{m}_{s}"] = int(t[(t.method == m) & (t.noise == s)].failed.sum())
    for s in [1, 3, 5]:
        x = t[(t.method == "Spline") & (t.noise == s)]
        K[f"spline_mean_excl_{s}"] = float(x[x.failed == 0].rmse.mean())
        for m in ORDER + ["ParametricLCDM"]:
            K[f"rmse_mean_{m}_{s}"] = float(t[(t.method == m) & (t.noise == s)].rmse.mean())
            K[f"rmse_med_{m}_{s}"] = float(t[(t.method == m) & (t.noise == s)].rmse.median())
    # paired tests between methods adjacent in the mean ranking
    L = []
    for s in [1, 3, 5]:
        means = {m: p.loc[s, m].mean() for m in ORDER}
        rank = sorted(ORDER, key=lambda m: means[m])
        for a, b in zip(rank[:-1], rank[1:]):
            d, lo, hi, pb, pw, frac = paired_realisation_test(p.loc[s, a].values, p.loc[s, b].values)
            L.append(f"{s} & {a} -- {b} & ${d:+.3f}$ & $[{lo:+.3f},{hi:+.3f}]$ & {pb:.3f} & {pw:.3f} & {frac:.2f} \\\\")
            K[f"pair_{s}_{a}_{b}"] = dict(diff=d, p_boot=pb, p_wil=pw, frac=frac)
        K[f"rank_{s}"] = rank
    w("tier1_pairs", L)
    # zt step-size, Om, GP coverage
    L = []
    for m in ORDER:
        x = t[(t.method == m) & (t.noise == 3)]
        cells = [f"{x[f'zt_h{h:g}'].median():.3f} ({x[f'zt_h{h:g}'].quantile(.84) - x[f'zt_h{h:g}'].quantile(.16):.3f})"
                 for h in [1e-4, 1e-3, 1e-2, 5e-2]]
        L.append(f"{m} & " + " & ".join(cells) + f" & {x.om_rmse.median():.3f} \\\\")
        for h in [1e-4, 1e-3, 1e-2, 5e-2]:
            K[f"zt_med_{m}_h{h:g}"] = float(x[f"zt_h{h:g}"].median())
        K[f"om_med_{m}"] = float(x.om_rmse.median())
        for s in [1, 3, 5]:
            K[f"zt_nan_{m}_{s}"] = int(t[(t.method == m) & (t.noise == s)].zt_hat.isna().sum())
    w("tier1_zt_step", L)
    for s in [1, 3, 5]:
        x = t[(t.method == "GP") & (t.noise == s)]
        K[f"gp_cov68_{s}"] = float(x.cov68.mean()); K[f"gp_cov95_{s}"] = float(x.cov95.mean())


def tier2():
    t = R("tier2_robust.csv")
    rows = [("flat_lcdm", "Hz", "Flat $\\Lambda$CDM, $H$"), ("nonflat_lcdm", "Hz", "Non-flat, $H$"),
            ("cpl_de", "Hz", "CPL, $H$"), ("feature", "Hz", "Local feature, $H$"),
            ("flat_lcdm", "DH", "Flat $\\Lambda$CDM, $D_H$"), ("nonflat_lcdm", "DH", "Non-flat, $D_H$"),
            ("cpl_de", "DH", "CPL, $D_H$")]
    L = []
    for c, o, lab in rows:
        sub = t[(t.cosmology == c) & (t.observable == o)]
        cells = [f"{sub[sub.method == m].rmse.median():.2f}" for m in ORDER]
        pf = sub[sub.method == "Param_flatLCDM"].rmse.median()
        pc = sub[sub.method == "Param_correct"].rmse.median() if (sub.method == "Param_correct").any() else np.nan
        means = {m: sub[sub.method == m].rmse.median() for m in ORDER}
        K[f"t2_meanrank_{c}_{o}"] = sorted(ORDER, key=lambda m: sub[sub.method == m].rmse.mean())
        K[f"t2_rank_{c}_{o}"] = sorted(ORDER, key=lambda m: means[m])
        for m in ORDER:
            K[f"t2_{c}_{o}_{m}"] = means[m]
        K[f"t2_{c}_{o}_pflat"] = pf; K[f"t2_{c}_{o}_pcorr"] = pc
        L.append(f"{lab} & " + " & ".join(cells) + f" & {pf:.2f} & " + ("--" if pd.isna(pc) else f"{pc:.2f}") + r" \\")
    w("tier2", L)
    # paired test in feature case
    sub = t[(t.cosmology == "feature")]
    p = sub.pivot_table(index="seed", columns="method", values="rmse")
    for a, b in [("GP", "Spline"), ("GP", "NN"), ("NN", "Spline"), ("GP", "Polynomial"), ("Spline", "Polynomial")]:
        d, lo, hi, pb, pw, frac = paired_realisation_test(p[a].values, p[b].values)
        K[f"feat_pair_{a}_{b}"] = dict(diff=d, p_boot=pb, p_wil=pw, frac=frac)


def real():
    s = R("real_loocv_summary.csv")
    meths = ["ParametricLCDM"] + ORDER + ["Nodal-range"]
    for name, sets in [("real_main", ["CC31", "BAO_legacy_v1", "BAO_DESI", "CC31+BAO_DESI"]),
                       ("real_sens", ["CC32", "CC34", "BAO_legacy", "CC31+BAO_legacy_v1"])]:
        L = []
        for m in meths:
            cells = []
            for d in sets:
                x = s[(s.dataset == d) & (s.method == m)].iloc[0]
                cells.append(f"{x.loocv_chi:.3f} & [{x.ci_lo:.2f}, {x.ci_hi:.2f}]" if x.loocv_chi < 100
                             else f"{x.loocv_chi:.3g} & [{x.ci_lo:.3g}, {x.ci_hi:.3g}]")
                K[f"real_{d}_{m}"] = dict(chi=float(x.loocv_chi), lo=float(x.ci_lo), hi=float(x.ci_hi), zt=float(x.zt_hat))
            L.append(f"{LAB.get(m, m)} & " + " & ".join(cells) + r" \\")
        w(name, L)
    for d in s.dataset.unique():
        b = s[(s.dataset == d) & (s.method == "_LCDM_bestfit")].iloc[0]
        K[f"lcdm_{d}"] = dict(H0=b.H0, H0_err=b.H0_err, Om=b.Om, Om_err=b.Om_err, chi2=b.chi2_fit, N=int(b.N))
    # rd sensitivity
    for d in ["CC31+BAO_DESI", "CC31+BAO_DESI_rd-2%", "CC31+BAO_DESI_rd+2%"]:
        for m in meths:
            x = s[(s.dataset == d) & (s.method == m)].iloc[0]
            K[f"rd_{d}_{m}"] = float(x.loocv_chi)
    pr = R("real_loocv_pairs.csv")
    for _, r in pr.iterrows():
        K[f"pair_{r.dataset}_{r.A}_{r.B}"] = dict(diff=r.chi_diff, p=r.p, p_holm=r.p_holm)
    L = []
    for d in ["CC31", "CC31+BAO_DESI"]:
        sub = pr[(pr.dataset == d) & ~pr.A.isin(["Nodal-range"]) & ~pr.B.isin(["Nodal-range"])].sort_values("p")
        for _, r in sub.head(6).iterrows():
            L.append(f"{d.replace('_', ' ')} & {r.A} -- {r.B} & ${r.chi_diff:+.3f}$ & {r.p:.3f} & {r.p_holm:.3f} \\\\")
    w("real_pairs", L)
    # calibration numbers: mean squared residual of the parametric fit
    f = R("real_loocv_folds.csv")
    from scipy.stats import kstest
    for d in ["CC31", "BAO_legacy_v1", "BAO_legacy", "BAO_DESI", "CC31+BAO_DESI"]:
        for m in ["GP", "ParametricLCDM"]:
            r = f[(f.dataset == d) & (f.method == m)].resid.dropna().values
            K[f"ks_{d}_{m}"] = float(kstest(r, "norm").pvalue)
            K[f"rstd_{d}_{m}"] = float(np.std(r))


def nodal():
    n = R("nodal_diagnostics.csv")
    L = []
    for d, lab in [("BAO_legacy_v1", "BAO v1 table"), ("BAO_legacy", "BAO legacy (corr.)"), ("BAO_desi", "BAO DESI DR2"),
                   ("CC31", "CC31"), ("CC31+BAO_desi", "CC31+DESI")]:
        for v in ["v1 [0,2] K8", "range K8", "range K4", "quantile K4"]:
            x = n[(n.dataset == d) & (n.variant == v)]
            if len(x) == 0:
                continue
            x = x.iloc[0]
            L.append(f"{lab} & {v.replace('[0,2]', '$[0,2]$')} & {x.chi:.3f} & {x.n_extrap} & {x.chi_interp_only:.3f} & {x.max_abs_resid:.1f} \\\\")
            K[f"nodal_{d}_{v}"] = dict(chi=x.chi, n_extrap=int(x.n_extrap), chi_in=x.chi_interp_only, maxr=x.max_abs_resid)
        L.append(r"\addlinespace")
    w("nodal", L[:-1])


def control():
    c = R("synthetic_control.csv"); pr = R("synthetic_control_pairs.csv")
    meths = ["ParametricLCDM"] + ORDER + ["Nodal-range"]
    L = []
    for d, lab in [("CC31", "CC31"), ("BAO_legacy_v1", "BAO v1"), ("BAO_legacy", "BAO legacy"),
                   ("BAO_desi", "BAO DESI"), ("CC31+BAO_desi", "CC31+DESI")]:
        sub = c[c.design == d]
        if len(sub) == 0:
            continue
        cells = []
        for m in meths:
            x = sub[sub.method == m]
            cells.append(f"{x.rmse.median():.2g} / {x.loocv_chi.median():.2f}")
            K[f"ctrl_{d}_{m}"] = dict(rmse_med=float(x.rmse.median()), chi_med=float(x.loocv_chi.median()),
                                      chi_lo=float(x.loocv_chi.quantile(.16)), chi_hi=float(x.loocv_chi.quantile(.84)))
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
        # rank agreement between RMSE and LOOCV within realisations (non-parametric methods)
        from scipy.stats import spearmanr
        rhos = []
        for seed, g in sub[sub.method.isin(ORDER)].groupby("seed"):
            rhos.append(spearmanr(g.rmse, g.loocv_chi).statistic)
        K[f"ctrl_rho_{d}"] = float(np.nanmedian(rhos))
        for b in ["Polynomial", "NN", "Spline", "Nodal", "Nodal-range"]:
            q = pr[(pr.design == d) & (pr.B == b)]
            K[f"ctrl_power_{d}_{b}"] = float(np.mean(q.p < 0.05))
            K[f"ctrl_gpbetter_{d}_{b}"] = float(np.mean(q.chi_diff < 0))
        K[f"ctrl_n_{d}"] = int(sub.seed.nunique())
    w("control", L)


def cov():
    c = R("covariance_loocv.csv")
    L = []
    order = ["GP (GLS)", "Polynomial (GLS)", "NN (GLS)", "GP (diag)", "Polynomial (diag)", "NN (diag)", "Spline (diag)", "Nodal (diag)"]
    for m in order:
        a = c[(c.hbar == "measured") & (c.method == m)].iloc[0]; b = c[(c.hbar == "model") & (c.method == m)].iloc[0]
        L.append(f"{m} & {a.weighting} & {a.loocv_chi:.3f} [{a.ci_lo:.2f}, {a.ci_hi:.2f}] & {b.loocv_chi:.3f} [{b.ci_lo:.2f}, {b.ci_hi:.2f}] \\\\")
        K[f"cov_{m}"] = dict(meas=a.loocv_chi, model=b.loocv_chi)
    w("covariance", L)


def nn_var():
    v = R("nn_seed_variability.csv"); t1 = R("tier1_main.csv")
    for s in [1, 3, 5]:
        base = t1[(t1.method == "NN") & (t1.noise == s)].set_index("seed").rmse
        allr = pd.concat([v[v.noise == s][["seed", "init_seed", "rmse"]],
                          pd.DataFrame(dict(seed=base.index, init_seed=42, rmse=base.values))])
        per_seed_mean = allr.groupby("init_seed").rmse.mean()
        within = allr.groupby("seed").rmse.std().mean()
        K[f"nnseed_{s}"] = dict(min=float(per_seed_mean.min()), max=float(per_seed_mean.max()),
                                within_real_sd=float(within), gp=float(t1[(t1.method == 'GP') & (t1.noise == s)].rmse.mean()),
                                poly=float(t1[(t1.method == 'Polynomial') & (t1.noise == s)].rmse.mean()))
    im = R("nn_impl_variability.csv")
    K["nnimpl"] = dict(torch=float(im.rmse_torch.mean()), numpy=float(im.rmse_numpy.mean()),
                       cross_mean=float(im.cross.mean()), cross_med=float(im.cross.median()),
                       seed0=im.iloc[0].to_dict())
    rc = R("nn_seed_real_cc.csv")
    K["nnseed_realcc"] = dict(min=float(rc.loocv_chi.min()), max=float(rc.loocv_chi.max()),
                              vals=rc.loocv_chi.round(3).tolist())


def tuned():
    tt = R("tier1_tuned.csv"); t1 = R("tier1_main.csv")
    L = []
    for m in ORDER:
        cells = []
        for s in [1, 3, 5]:
            x = tt[(tt.method == m) & (tt.noise == s)]
            b0 = t1[(t1.method == m) & (t1.noise == s)]
            b = b0[b0.seed.isin(x.seed)].rmse   # baseline on the SAME realisations
            mode = x.chosen.mode().iloc[0]
            cells.append(f"{b.mean():.3f} & {x.rmse.mean():.3f} & {mode:g}")
            K[f"tuned_{m}_{s}"] = dict(base=float(b.mean()), tuned=float(x.rmse.mean()), mode=float(mode))
        L.append(f"{m} & " + " & ".join(cells) + r" \\")
    w("tuned", L)
    for s in [1, 3, 5]:
        means = {m: tt[(tt.method == m) & (tt.noise == s)].rmse.mean() for m in ORDER}
        K[f"tuned_rank_{s}"] = sorted(ORDER, key=lambda m: means[m])
        p = tt[tt.noise == s].pivot_table(index="seed", columns="method", values="rmse")
        r = K[f"tuned_rank_{s}"]
        d, lo, hi, pb, pw, frac = paired_realisation_test(p[r[0]].values, p[r[1]].values)
        K[f"tuned_pair_{s}"] = dict(a=r[0], b=r[1], diff=d, p_boot=pb, p_wil=pw)


def data_tables():
    L = []
    for k, lab in [("legacy", "legacy")]:
        z, H, s, src = load_bao_data(k)
        for a, b, c, d in zip(z, H, s, src):
            L.append(f"{a:.2f} & ${b:.2f}\\pm{c:.2f}$ & {d.replace('&', '\\&')} \\\\")
    w("bao_legacy", L)
    z, x, sx, src = desi_dh_over_rd()
    zz, H, s, _ = load_bao_data("desi")
    w("bao_desi", [f"{a:.3f} & ${b:.3f}\\pm{c:.3f}$ & ${h:.2f}\\pm{e:.2f}$ & {d} \\\\"
                   for a, b, c, h, e, d in zip(z, x, sx, H, s, src)])


if __name__ == "__main__":
    for f in (tier1, tier2, real, nodal, control, cov, nn_var, tuned, data_tables):
        try:
            f()
        except FileNotFoundError as e:
            log("skip", f.__name__, e)
    json.dump(K, open("results/key_numbers.json", "w"), indent=1, default=float)
