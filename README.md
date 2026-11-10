# Non-parametric H(z) reconstruction benchmark

Code, data tables and results for

> D. Mukherjee, *The Limits of Synthetic Validation: A Real-Data-Confronted Benchmark of Non-Parametric $H(z)$ Reconstruction*).

Everything in the paper — every table, figure and number — is produced by the scripts in this repository from fixed random seeds.

## Contents

```
hzbench/                 library
  cosmology.py           fiducial expansion histories (flat/non-flat LCDM, CPL, localised feature)
  data_gen.py            synthetic H(z) and D_H data; generation at a given (real) redshift design
  methods.py             the five reconstruction implementations (NN, polynomial, spline, GP, nodal)
                         and the variants used in the revision (nodal binning options, NN seeds, ...)
  metrics.py             RMSE, q(z), z_t, Om(z), parametric reference fits
  loocv.py               leave-one-out cross-validation with fold-by-fold output
  bootstrap_stats.py     empirical bootstrap intervals, paired tests (real data and synthetic)
  covariance.py          approximate CC systematic covariance and GLS fits
  real_data.py           CC31 / CC32 / CC34 compilations (with sources)
  real_bao_data.py       radial-BAO sets: v1 table (reproducibility only), corrected legacy set,
                         DESI DR2 D_H/r_d converted with r_d = 147.09 Mpc (with sources)
scripts/
  run_all.sh             runs everything below, then figures and tables
  run_tier1.py           Tier-1 synthetic benchmark + noiseless bias, q(z) step sizes, Om(z), GP coverage
  run_tier2.py           Tier-2 grid (3 histories x 2 observables) + localised-feature history
  run_tuned.py           common 5-fold-CV hyperparameter tuning of all five methods
  run_nn_variability.py  NN initialisation-seed and implementation variability
  run_real_data.py       LOOCV on all real data sets, all methods, fold-by-fold residuals, all pairwise tests
  run_nodal_diagnostics.py  nodal binning variants, per-point residuals, extrapolation flags
  run_synthetic_control.py  synthetic data at the real redshifts/uncertainties, scored by RMSE and LOOCV
  run_covariance_gls.py  CC covariance-sensitivity analysis
  make_figures.py        all figures -> figures/
  make_tables.py         LaTeX table bodies -> results/tables/, numbers -> results/key_numbers.json
results/                 all CSV outputs of the published run (committed)
figures/                 all figures of the paper (PDF)
```

## Reproducing

```bash
pip install -r requirements.txt
bash scripts/run_all.sh            # ~1.5-2 h on one CPU core
HZ_FAST=1 bash scripts/run_all.sh  # ~5 min smoke test with very few realisations
```

Individual steps can be run on their own, e.g. `python scripts/run_real_data.py`.

## Seeds and conventions

* Synthetic Tier-1/Tier-2 realisation *k* (k = 0, 1, ...) uses `numpy.random.default_rng(k)` for both the redshift draw and the noise.
* Realisation counts: Tier 1 30 per noise level; Tier 2 10 per cell (30 for the localised feature); synthetic control 20 (CC31),
  30 (each BAO design), 10 (CC31+DESI); NN seed/implementation tests on the first 10 Tier-1 realisations; common tuning on the
  first 6 Tier-1 realisations per noise level.
* Synthetic-control realisation *k* uses seed `10000 + k`; the common-tuning CV folds of realisation *k* use seed `1000 + k`.
* NN weights are initialised with `torch.manual_seed(42)` unless stated otherwise (seeds 0–4 in `run_nn_variability.py`).
* GP optimiser restarts use `random_state=0`.  (In v1 this was unset, which made GP results non-deterministic; this is why the
  GP Tier-1 mean at sigma = 3 changed from 1.076 to 1.148 km/s/Mpc.)
* Bootstrap: B = 2000 resamples, seed 0.
* LOOCV residual convention: r_i = (Hhat_{-i}(z_i) - H_i) / sigma_i.  `results/real_loocv_folds.csv` lists every held-out point.

Results were produced with Python 3.12, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, PyTorch 2.15 (CPU).

## Data provenance

All data values were checked against the primary publications during the revision; see the docstrings of
`hzbench/real_data.py` and `hzbench/real_bao_data.py` and Table 5 of the paper.  The 10-point BAO table released with v1 contained
mislabelled entries (documented in `real_bao_data.py`) and is retained only so that the v1 numbers can be reproduced.

## Licence

MIT (see `LICENSE`).
