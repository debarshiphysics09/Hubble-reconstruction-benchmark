"""
NN variability over weight-initialisation seeds and implementations.

(a) The first 10 Tier-1 realisations (sigma = 1, 3, 5) refit with init seeds
    {0, 1, 2, 3, 4} in addition to the baseline seed 42 (in tier1_main.csv).
(b) Independent NumPy implementation (same architecture, loss, optimiser,
    hyperparameters; different initialiser) on the same realisations at
    sigma = 3 (first 10 realisations), with curve-to-curve RMSE against the PyTorch fit.
(c) Real CC31 LOOCV score for the six init seeds.
Outputs: results/nn_seed_variability.csv, results/nn_impl_variability.csv,
         results/nn_seed_real_cc.csv
"""
from _common import *
import pandas as pd
from hzbench.cosmology import FLAT_LCDM
from hzbench.data_gen import generate_Hz_data
from hzbench.metrics import rmse
from hzbench.loocv import loocv_residuals
from hzbench.real_data import load_cc_data

ZGRID = np.linspace(0, 2, 400); HT = FLAT_LCDM.Hz(ZGRID)
SEEDS = [0, 1] if FAST else [0, 1, 2, 3, 4]
NREAL = 2 if FAST else 10
rows = []
for s in [1.0, 3.0, 5.0]:
    for seed in range(NREAL):
        z, H, sig, _ = generate_Hz_data(FLAT_LCDM, n=80, sigma=s, seed=seed)
        for k in SEEDS:
            rows.append(dict(noise=s, seed=seed, init_seed=k,
                             rmse=rmse(M.fit_nn(z, H, sig, init_seed=k)(ZGRID), HT)))
    log(f"[nn seeds] sigma={s} done")
pd.DataFrame(rows).to_csv("results/nn_seed_variability.csv", index=False)

rows = []
for seed in range(NREAL):
    z, H, sig, _ = generate_Hz_data(FLAT_LCDM, n=80, sigma=3.0, seed=seed)
    pt, pn = M.fit_nn(z, H, sig)(ZGRID), M.fit_nn_numpy(z, H, sig)(ZGRID)
    rows.append(dict(seed=seed, rmse_torch=rmse(pt, HT), rmse_numpy=rmse(pn, HT), cross=rmse(pt, pn)))
log("[nn impl] done")
pd.DataFrame(rows).to_csv("results/nn_impl_variability.csv", index=False)

z, H, sig = load_cc_data("CC31")
rows = []
for k in [42] + SEEDS:
    chi, r, _ = loocv_residuals(lambda a, b, c: M.fit_nn(a, b, c, init_seed=k), z, H, sig)
    rows.append(dict(init_seed=k, loocv_chi=chi))
    log(f"[nn real cc] seed {k}: chi={chi:.3f}")
pd.DataFrame(rows).to_csv("results/nn_seed_real_cc.csv", index=False)
