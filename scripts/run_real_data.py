"""
LOOCV on real compilations, all methods, with fold-by-fold residuals.

Data sets:  CC31 (baseline), CC32, CC34 (sensitivity), BAO legacy_v1 (v1
table as released, reproducibility only), BAO legacy (corrected, N=7),
BAO DESI DR2 (N=6, r_d = 147.09 Mpc), and the unions CC31+legacy_v1 (v1),
CC31+DESI, CC32+DESI.  DESI unions are also re-run with r_d shifted by
+/-2 per cent to show the conditional dependence on the sound horizon.

Methods: the five baseline implementations (Nodal = v1 prescription, bins
on [0,2]), "Nodal-range" (eight equal-width bins spanning the redshift range
of the compilation being analysed), and the flat-LCDM parametric fit.

Outputs: results/real_loocv_summary.csv   (chi, 68% interval, z_t)
         results/real_loocv_folds.csv     (every held-out point, every method)
         results/real_loocv_pairs.csv     (all paired comparisons, Holm-adjusted)
"""
from _common import *
import pandas as pd
from hzbench.real_data import load_cc_data
from hzbench.real_bao_data import load_bao_data, RD_PLANCK
from hzbench.loocv import loocv_residuals, parametric_fitter
from hzbench.bootstrap_stats import bootstrap_chi_ci, all_pairs
from hzbench.metrics import deceleration_parameter, transition_redshift_from_curve, parametric_fit


def bao(name, rd=RD_PLANCK):
    z, H, s, _ = load_bao_data(name, rd=rd)
    return z, H, s, ["BAO"] * len(z)


def cc(name):
    z, H, s = load_cc_data(name)
    return z, H, s, ["CC"] * len(z)


def union(a, b):
    z = np.concatenate([a[0], b[0]]); o = np.argsort(z, kind="stable")
    return (z[o], np.concatenate([a[1], b[1]])[o], np.concatenate([a[2], b[2]])[o],
            list(np.array(a[3] + b[3])[o]))


DATASETS = {
    "CC31": cc("CC31"), "CC32": cc("CC32"), "CC34": cc("CC34"),
    "BAO_legacy_v1": bao("legacy_v1"), "BAO_legacy": bao("legacy"), "BAO_DESI": bao("desi"),
    "CC31+BAO_legacy_v1": union(cc("CC31"), bao("legacy_v1")),
    "CC31+BAO_DESI": union(cc("CC31"), bao("desi")),
    "CC32+BAO_DESI": union(cc("CC32"), bao("desi")),
    "CC31+BAO_DESI_rd-2%": union(cc("CC31"), bao("desi", RD_PLANCK * 0.98)),
    "CC31+BAO_DESI_rd+2%": union(cc("CC31"), bao("desi", RD_PLANCK * 1.02)),
}
if FAST:
    DATASETS = {k: DATASETS[k] for k in ["BAO_legacy_v1", "BAO_DESI"]}

summ, folds, pairs = [], [], []
for dname, (z, H, s, kind) in DATASETS.items():
    zr = (z.min(), z.max())
    fit = dict(FITTERS)
    fit["Nodal-range"] = lambda a, b, c, zr=zr: M.fit_nodal(a, b, c, zmin=zr[0], zmax=zr[1] + 1e-9)
    fit["ParametricLCDM"] = parametric_fitter("flat_lcdm")
    zg = np.linspace(zr[0], zr[1], 400)
    resid = {}
    for name, f in fit.items():
        chi, r, pred = loocv_residuals(f, z, H, s)
        resid[name] = r
        _, lo, hi = bootstrap_chi_ci(r)
        try:
            zt = transition_redshift_from_curve(zg, deceleration_parameter(f(z, H, s), zg))
        except Exception:
            zt = np.nan
        summ.append(dict(dataset=dname, N=len(z), method=name, loocv_chi=chi, ci_lo=lo, ci_hi=hi,
                         zt_hat=zt, mean_resid=float(np.nanmean(r)), chi2_per_N=chi ** 2))
        for i in range(len(z)):
            folds.append(dict(dataset=dname, method=name, i=i, kind=kind[i], z=z[i], H=H[i],
                              sigma=s[i], pred=pred[i], resid=r[i]))
        log(f"  {dname:22s} {name:15s} chi={chi:6.3f} [{lo:.3f},{hi:.3f}] zt={zt:.3f}")
    for a, b, d, p, ph in all_pairs(resid):
        pairs.append(dict(dataset=dname, A=a, B=b, chi_diff=d, p=p, p_holm=ph))
    _, popt, pcov = parametric_fit(z, H, s)
    summ.append(dict(dataset=dname, N=len(z), method="_LCDM_bestfit", H0=popt[0], H0_err=np.sqrt(pcov[0, 0]),
                     Om=popt[1], Om_err=np.sqrt(pcov[1, 1]),
                     chi2_fit=float(np.sum(((parametric_fit(z, H, s)[0](z) - H) / s) ** 2))))

pd.DataFrame(summ).to_csv("results/real_loocv_summary.csv", index=False)
pd.DataFrame(folds).to_csv("results/real_loocv_folds.csv", index=False)
pd.DataFrame(pairs).to_csv("results/real_loocv_pairs.csv", index=False)
