"""
Covariance-sensitivity LOOCV on CC31 (see hzbench/covariance.py).
Output: results/covariance_loocv.csv
"""
from _common import *
import pandas as pd
from hzbench.real_data import load_cc_data
from hzbench.covariance import build_covariance, fit_poly_gls, fit_gp_gls, fit_nn_gls
from hzbench.metrics import parametric_fit
from hzbench.bootstrap_stats import bootstrap_chi_ci

z, H, s = load_cc_data("CC31")
rows = []
for hbar_mode in ("measured", "model"):
    n = len(z); out = {}
    gls = {"GP (GLS)": fit_gp_gls, "Polynomial (GLS)": fit_poly_gls, "NN (GLS)": fit_nn_gls}
    diag = {"Spline (diag)": FITTERS["Spline"], "Nodal (diag)": FITTERS["Nodal"],
            "GP (diag)": FITTERS["GP"], "Polynomial (diag)": FITTERS["Polynomial"], "NN (diag)": FITTERS["NN"]}
    res = {k: np.full(n, np.nan) for k in list(gls) + list(diag)}
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        if hbar_mode == "measured":
            Hbar = H
        else:
            pf = parametric_fit(z[m], H[m], s[m])[0]; Hbar = pf(z)
        Cov = build_covariance(z, Hbar, s)
        sd = np.sqrt(np.diag(Cov))
        Csub = Cov[np.ix_(m, m)]
        for k, f in gls.items():
            res[k][i] = (f(z[m], H[m], Csub)(np.array([z[i]]))[0] - H[i]) / sd[i]
        for k, f in diag.items():
            res[k][i] = (f(z[m], H[m], sd[m])(np.array([z[i]]))[0] - H[i]) / sd[i]
    for k, r in res.items():
        chi, lo, hi = bootstrap_chi_ci(r)
        rows.append(dict(hbar=hbar_mode, method=k, weighting="full covariance" if "GLS" in k else "diagonal, $\\sqrt{C_{ii}}$",
                         loocv_chi=chi, ci_lo=lo, ci_hi=hi))
        log(f"[cov {hbar_mode}] {k:18s} chi={chi:.3f} [{lo:.3f},{hi:.3f}]")
pd.DataFrame(rows).to_csv("results/covariance_loocv.csv", index=False)
