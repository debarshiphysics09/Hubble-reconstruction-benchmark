"""
Tier 2 robustness grid: 3 smooth fiducial histories x 2 observables
(direct H(z) at sigma=3; BAO D_H proxy at 3%), 10 realisations each, plus
(revision) the localised-feature history at sigma=3 with 30 realisations.

Parametric references: the 2-parameter flat-LCDM fit (correct only for the
flat-LCDM truth; MISSPECIFIED otherwise) and, for non-flat and CPL truths,
the correctly specified 3- and 4-parameter family.
Output: results/tier2_robust.csv
"""
from _common import *
import pandas as pd
from hzbench.cosmology import ALL_COSMOLOGIES, FEATURE
from hzbench.data_gen import generate_Hz_data, generate_DH_data
from hzbench.metrics import rmse, parametric_reference_rmse

ZGRID = np.linspace(0.0, 2.0, 400)
N_ROB = 2 if FAST else 10
N_FEAT = 3 if FAST else 30
rows = []
configs = [(c, o, N_ROB) for c in ALL_COSMOLOGIES for o in ("Hz", "DH")] + [("feature", "Hz", N_FEAT)]
for cname, obs, nseed in configs:
    cosmo = FEATURE if cname == "feature" else ALL_COSMOLOGIES[cname]
    Ht = cosmo.Hz(ZGRID)
    for seed in range(nseed):
        if obs == "Hz":
            z, H, s, _ = generate_Hz_data(cosmo, n=80, sigma=3.0, seed=seed)
        else:
            z, H, s, _ = generate_DH_data(cosmo, n=80, rel_sigma=0.03, seed=seed)
        base = dict(cosmology=cname, observable=obs, seed=seed)
        for name in ORDER:
            r = rmse(FITTERS[name](z, H, s)(ZGRID), Ht)
            rows.append({**base, "method": name, "rmse": r, "failed": int(r > 20)})
        r, _ = parametric_reference_rmse(z, H, s, cosmo, ZGRID, "flat_lcdm")
        rows.append({**base, "method": "Param_flatLCDM", "rmse": r})
        if cname in ("nonflat_lcdm", "cpl_de"):
            r, _ = parametric_reference_rmse(z, H, s, cosmo, ZGRID, cname)
            rows.append({**base, "method": "Param_correct", "rmse": r})
        elif cname == "flat_lcdm":
            rows.append({**base, "method": "Param_correct", "rmse": rows[-1]["rmse"]})
    log(f"[tier2] {cname}/{obs} done")
pd.DataFrame(rows).to_csv("results/tier2_robust.csv", index=False)
