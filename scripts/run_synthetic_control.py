"""
Synthetic control at the REAL redshifts and uncertainties (referee 2 points
2 and 5).  Flat-LCDM truth (H0=70, Om=0.3); 20 realisations for CC31, 30 for each
BAO design and 10 for CC31+BAO DESI.  For each real design (CC31,
BAO legacy_v1, BAO legacy, BAO DESI, CC31+BAO DESI) we draw Gaussian noise
with exactly the quoted sigma_i, and score every method on the SAME
realisation by
   (i)  RMSE against the known truth on the design's redshift range, and
   (ii) the LOOCV score used on real data.
This separates the effect of sampling/noise (design) from that of real vs
synthetic data, gives the distribution of the LOOCV score expected when the
errors are correctly calibrated, and estimates the power of the paired
bootstrap at the real sample sizes.
Outputs: results/synthetic_control.csv (per realisation x method),
         results/synthetic_control_pairs.csv, results/synthetic_control_resid.csv
"""
from _common import *
import pandas as pd
from hzbench.cosmology import FLAT_LCDM
from hzbench.data_gen import generate_at_design
from hzbench.real_data import load_cc_data
from hzbench.real_bao_data import load_bao_data
from hzbench.loocv import loocv_residuals, parametric_fitter
from hzbench.metrics import rmse
from hzbench.bootstrap_stats import paired_bootstrap_pvalue

designs = {}
z, H, s = load_cc_data("CC31"); designs["CC31"] = (z, s, 20)
for k in ("legacy_v1", "legacy", "desi"):
    zb, Hb, sb, _ = load_bao_data(k); designs["BAO_" + k] = (zb, sb, 30)
zb, Hb, sb, _ = load_bao_data("desi")
zz = np.concatenate([z, zb]); o = np.argsort(zz)
designs["CC31+BAO_desi"] = (zz[o], np.concatenate([s, sb])[o], 10)
ONLY = os.environ.get("HZ_DESIGNS")
if ONLY:
    designs = {k: v for k, v in designs.items() if k in ONLY.split(",")}
if FAST:
    designs = {k: (a, b, 2) for k, (a, b, c) in designs.items() if k.startswith("BAO_l")}

OUT = os.environ.get("HZ_CONTROL_SUFFIX", "")
rows, prow, rrow = [], [], []
for dname, (zd, sd, nreal) in designs.items():
    zr = (zd.min(), zd.max())
    zg = np.linspace(zr[0], zr[1], 400); Ht = FLAT_LCDM.Hz(zg)
    fit = dict(FITTERS)
    fit["Nodal-range"] = lambda a, b, c, zr=zr: M.fit_nodal(a, b, c, zmin=zr[0], zmax=zr[1] + 1e-9)
    fit["ParametricLCDM"] = parametric_fitter("flat_lcdm")
    for seed in range(nreal):
        zz_, Hobs, ss, _ = generate_at_design(FLAT_LCDM, zd, sd, seed=10_000 + seed)
        res = {}
        for name, f in fit.items():
            try:
                r_ = rmse(f(zz_, Hobs, ss)(zg), Ht)
            except Exception:
                r_ = np.nan
            chi, r, _ = loocv_residuals(f, zz_, Hobs, ss)
            res[name] = r
            rows.append(dict(design=dname, seed=seed, method=name, rmse=r_, loocv_chi=chi))
            for i, v in enumerate(r):
                rrow.append(dict(design=dname, seed=seed, method=name, i=i, resid=v))
        for other in ["Polynomial", "NN", "Spline", "Nodal", "Nodal-range"]:
            d, p = paired_bootstrap_pvalue(res["GP"], res[other])
            prow.append(dict(design=dname, seed=seed, A="GP", B=other, chi_diff=d, p=p))
        log(f"[control] {dname} realisation {seed} done")
    pd.DataFrame(rows).to_csv(f"results/synthetic_control{OUT}.csv", index=False)
    pd.DataFrame(prow).to_csv(f"results/synthetic_control_pairs{OUT}.csv", index=False)
    pd.DataFrame(rrow).to_csv(f"results/synthetic_control_resid{OUT}.csv", index=False)
