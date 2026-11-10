"""
Tier 1 synthetic benchmark (flat LCDM, direct H(z), N=80, sigma in {1,3,5},
30 noise realisations; data seed = realisation index 0..29) plus the
diagnostics added in revision:

  * q(z) and z_t for finite-difference steps h in {1e-4, 1e-3, 1e-2, 5e-2}
  * Om(z) diagnostic RMSE on 0.1 <= z <= 2
  * GP 68%/95% band coverage of the true curve on the grid
  * noiseless-bias run: same redshift draws, H_obs = H_true (sigma used
    only as weights), giving each method's approximation bias floor.

Outputs: results/tier1_main.csv, results/tier1_noiseless.csv
"""
from _common import *
import time
import pandas as pd
from hzbench.cosmology import FLAT_LCDM
from hzbench.data_gen import generate_Hz_data
from hzbench.metrics import (rmse, deceleration_parameter, transition_redshift_from_curve,
                             om_diagnostic, parametric_reference_rmse)

ZGRID = np.linspace(0.0, 2.0, 400)
ZQ = np.linspace(0.05, 1.95, 300)          # q(z) comparison grid (avoids edges)
ZOM = np.linspace(0.1, 2.0, 300)
STEPS = [1e-4, 1e-3, 1e-2, 5e-2]
NOISE = [1.0, 3.0, 5.0]
NSEED = 3 if FAST else 30
H_TRUE = FLAT_LCDM.Hz(ZGRID)
Q_TRUE = FLAT_LCDM.q(ZQ)
ZT_TRUE = FLAT_LCDM.transition_redshift()


def evaluate(name, pred_fn, sigma_lvl, seed, extra=None):
    row = dict(noise=sigma_lvl, seed=seed, method=name)
    row["rmse"] = rmse(pred_fn(ZGRID), H_TRUE)
    for h in STEPS:
        q = deceleration_parameter(pred_fn, ZQ, h=h)
        row[f"zt_h{h:g}"] = transition_redshift_from_curve(ZQ, q)
        row[f"qrmse_h{h:g}"] = rmse(q, Q_TRUE)
    row["zt_hat"] = row["zt_h0.001"]
    row["om_rmse"] = rmse(om_diagnostic(pred_fn, ZOM), np.full_like(ZOM, FLAT_LCDM.Om))
    row["failed"] = int(row["rmse"] > 20.0)
    if extra:
        row.update(extra)
    return row


def run(noiseless=False):
    rows = []
    for s in NOISE:
        t0 = time.time()
        for seed in range(NSEED):
            z, H_obs, sig, H_true = generate_Hz_data(FLAT_LCDM, n=80, sigma=s, seed=seed)
            if noiseless:
                H_obs = H_true.copy()
            for name in ORDER:
                if noiseless and name == "NN" and s != NOISE[0]:
                    # NN weights are uniform (sigma constant) -> identical fit at every sigma
                    prev = [r for r in rows if r["method"] == "NN" and r["seed"] == seed and r["noise"] == NOISE[0]][0]
                    rows.append({**prev, "noise": s}); continue
                if name == "GP":
                    pred, pstd, _ = M.fit_gp(z, H_obs, sig, return_model=True)
                    m, sd = pstd(ZGRID)
                    extra = dict(cov68=float(np.mean(np.abs(m - H_TRUE) <= sd)),
                                 cov95=float(np.mean(np.abs(m - H_TRUE) <= 1.96 * sd)))
                    rows.append(evaluate(name, pred, s, seed, extra))
                else:
                    rows.append(evaluate(name, FITTERS[name](z, H_obs, sig), s, seed))
            if not noiseless:
                r, _ = parametric_reference_rmse(z, H_obs, sig, FLAT_LCDM, ZGRID)
                rows.append(dict(noise=s, seed=seed, method="ParametricLCDM", rmse=r))
        log(f"[tier1{' noiseless' if noiseless else ''}] sigma={s} done in {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows); df["zt_true"] = ZT_TRUE
    return df


if __name__ == "__main__":
    run(False).to_csv("results/tier1_main.csv", index=False)
    run(True).to_csv("results/tier1_noiseless.csv", index=False)
