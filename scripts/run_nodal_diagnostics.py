"""
Nodal-estimator diagnostics (referee 1 point 2, referee 2 point 1).

For each data set, LOOCV of the nodal estimator under several binning
prescriptions, with per-point residuals and the location of every held-out
point relative to the populated nodes (inside the node range = interpolated;
outside = constant extrapolation).

  v1        K=8 equal-width bins on [0, 2]         (prescription of v1)
  0-2.4     K=8 equal-width bins on [0, 2.4]
  range-K8  K=8 equal-width bins on [z_min, z_max] of the data set
  range-K6, range-K4   fewer bins on the data range
  quant-K4  4 equal-count (density-adaptive) bins, nodes at weighted mean z
  quant-K8  min(8, N) equal-count bins
Output: results/nodal_diagnostics.csv, results/nodal_diagnostics_points.csv
"""
from _common import *
import pandas as pd
from hzbench.real_data import load_cc_data
from hzbench.real_bao_data import load_bao_data
from hzbench.bootstrap_stats import bootstrap_chi_ci


def sets():
    out = {}
    for k in ("legacy_v1", "legacy", "desi"):
        z, H, s, _ = load_bao_data(k); out["BAO_" + k] = (z, H, s)
    z, H, s = load_cc_data("CC31"); out["CC31"] = (z, H, s)
    zb, Hb, sb, _ = load_bao_data("desi")
    zz = np.concatenate([z, zb]); o = np.argsort(zz)
    out["CC31+BAO_DESI"] = (zz[o], np.concatenate([H, Hb])[o], np.concatenate([s, sb])[o])
    zb, Hb, sb, _ = load_bao_data("legacy_v1")
    zz = np.concatenate([z, zb]); o = np.argsort(zz)
    out["CC31+BAO_legacy_v1"] = (zz[o], np.concatenate([H, Hb])[o], np.concatenate([s, sb])[o])
    return out


rows, pts = [], []
for dname, (z, H, s) in sets().items():
    zr = (z.min(), z.max() + 1e-9)
    variants = {
        "v1 [0,2] K8": dict(n_bins=8, zmin=0, zmax=2),
        "[0,2.4] K8": dict(n_bins=8, zmin=0, zmax=2.4),
        "range K8": dict(n_bins=8, zmin=zr[0], zmax=zr[1]),
        "range K6": dict(n_bins=6, zmin=zr[0], zmax=zr[1]),
        "range K4": dict(n_bins=4, zmin=zr[0], zmax=zr[1]),
        "quantile K4": dict(n_bins=4, binning="quantile"),
        "quantile K8": dict(n_bins=8, binning="quantile"),
    }
    for vname, opt in variants.items():
        n = len(z); r = np.full(n, np.nan); extrap = np.zeros(n, bool)
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            f, c, v = M.fit_nodal(z[m], H[m], s[m], return_nodes=True, **opt)
            r[i] = (f(np.array([z[i]]))[0] - H[i]) / s[i]
            extrap[i] = (z[i] < c.min()) or (z[i] > c.max())
            pts.append(dict(dataset=dname, variant=vname, z=z[i], resid=r[i], extrapolated=bool(extrap[i])))
        chi, lo, hi = bootstrap_chi_ci(r)
        chi_in = float(np.sqrt(np.mean(r[~extrap] ** 2))) if (~extrap).any() else np.nan
        rows.append(dict(dataset=dname, variant=vname, N=n, chi=chi, ci_lo=lo, ci_hi=hi,
                         n_extrap=int(extrap.sum()), chi_interp_only=chi_in,
                         max_abs_resid=float(np.max(np.abs(r)))))
        log(f"{dname:20s} {vname:12s} chi={chi:6.3f} n_extrap={extrap.sum()} chi_interp={chi_in:.3f}")
pd.DataFrame(rows).to_csv("results/nodal_diagnostics.csv", index=False)
pd.DataFrame(pts).to_csv("results/nodal_diagnostics_points.csv", index=False)
