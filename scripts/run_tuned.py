"""
Common hyperparameter-tuning procedure (referee 2 point 8).

Every method selects ONE complexity hyperparameter from a small grid by
5-fold cross-validation on the training realisation (score = mean squared
standardised prediction error); the selected value is then used to fit all
80 points.  Grids:
   Polynomial  degree            {2, 3, 4, 5, 6}
   Spline      s / N             {0.25, 0.5, 1, 2, 4}
   Nodal       bins K on [0, 2]  {4, 6, 8, 10, 12}
   GP          length scale      {0.25, 0.5, 1, 2, 4} (amplitude by ML-II at fixed length)
   NN          weight decay      {1e-5, 1e-4, 1e-3, 1e-2}
The first 6 of the 30 Tier-1 realisations per noise level (seeds 0-5).
Output: results/tier1_tuned.csv
"""
from _common import *
import pandas as pd
from hzbench.cosmology import FLAT_LCDM
from hzbench.data_gen import generate_Hz_data
from hzbench.metrics import rmse

ZGRID = np.linspace(0, 2, 400); HT = FLAT_LCDM.Hz(ZGRID)
NREAL = 2 if FAST else 6
GRIDS = {
    "Polynomial": ("degree", [2, 3, 4, 5, 6], lambda v: (lambda z, H, s: M.fit_poly(z, H, s, degree=v))),
    "Spline": ("s_factor", [0.25, 0.5, 1, 2, 4], lambda v: (lambda z, H, s: M.fit_spline(z, H, s, s_factor=v))),
    "Nodal": ("n_bins", [4, 6, 8, 10, 12], lambda v: (lambda z, H, s: M.fit_nodal(z, H, s, n_bins=v))),
    "GP": ("length_scale", [0.25, 0.5, 1, 2, 4], lambda v: (lambda z, H, s: M.fit_gp(z, H, s, length_scale=v))),
    "NN": ("weight_decay", [1e-5, 1e-4, 1e-3, 1e-2], lambda v: (lambda z, H, s: M.fit_nn(z, H, s, weight_decay=v))),
}


def cv_score(fitter, z, H, s, folds):
    err = []
    for k in range(5):
        te = folds == k; tr = ~te
        try:
            p = fitter(z[tr], H[tr], s[tr])(z[te])
            err.append(((p - H[te]) / s[te]) ** 2)
        except Exception:
            return np.inf
    e = np.concatenate(err)
    return float(np.mean(e)) if np.all(np.isfinite(e)) else np.inf


rows = []
for sl in [1.0, 3.0, 5.0]:
    for seed in range(NREAL):
        z, H, s, _ = generate_Hz_data(FLAT_LCDM, n=80, sigma=sl, seed=seed)
        folds = np.random.default_rng(1000 + seed).permutation(np.arange(len(z)) % 5)
        for name, (pname, grid, make) in GRIDS.items():
            scores = [cv_score(make(v), z, H, s, folds) for v in grid]
            best = grid[int(np.argmin(scores))]
            r = rmse(make(best)(z, H, s)(ZGRID), HT)
            rows.append(dict(noise=sl, seed=seed, method=name, param=pname, chosen=best, rmse=r))
        log(f"[tuned] sigma={sl} seed={seed}")
    pd.DataFrame(rows).to_csv("results/tier1_tuned.csv", index=False)
