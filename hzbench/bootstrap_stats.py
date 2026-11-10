"""
Empirical bootstrap summaries of LOOCV residuals, and paired tests.

The LOOCV score of a method on N points is chi = sqrt(mean(r_i^2)) with
r_i = (Hhat_{-i}(z_i) - H_i)/sigma_i.  We resample the N held-out POINTS
with replacement (B=2000), recompute chi from the already-computed
residuals, and report the 16th-84th percentile range ("68% interval").
For two methods, the same resampled indices are used for both, giving the
bootstrap distribution of chi_A - chi_B; the two-sided p-value is twice the
fraction of resampled differences with the sign opposite to the observed
difference.  The leave-one-out fits are NOT repeated inside the bootstrap,
and LOOCV residuals are not independent (each pair of folds shares N-2
training points), so these numbers are DESCRIPTIVE summaries of the spread
of the residuals, not frequentist tests with guaranteed size.
"""
import numpy as np
from itertools import combinations


def bootstrap_chi_ci(resid, B=2000, seed=0, ci=(16, 84)):
    rng = np.random.default_rng(seed)
    resid = np.asarray(resid)[~np.isnan(resid)]
    n = len(resid)
    chi_obs = float(np.sqrt(np.mean(resid ** 2)))
    idx = rng.integers(0, n, (B, n))
    boots = np.sqrt(np.mean(resid[idx] ** 2, axis=1))
    lo, hi = np.percentile(boots, ci)
    return chi_obs, float(lo), float(hi)


def paired_bootstrap_pvalue(resid_a, resid_b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(resid_a), np.asarray(resid_b)
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[m], b[m]
    n = len(a)
    obs = np.sqrt(np.mean(a ** 2)) - np.sqrt(np.mean(b ** 2))
    idx = rng.integers(0, n, (B, n))
    boots = np.sqrt(np.mean(a[idx] ** 2, axis=1)) - np.sqrt(np.mean(b[idx] ** 2, axis=1))
    p = 2 * np.mean(boots <= 0) if obs >= 0 else 2 * np.mean(boots >= 0)
    return float(obs), float(min(p, 1.0))


def all_pairs(residuals, B=2000, seed=0):
    """residuals: dict name -> array.  Returns list of (A, B, diff, p, p_holm)."""
    names = list(residuals)
    rows = []
    for a, b in combinations(names, 2):
        d, p = paired_bootstrap_pvalue(residuals[a], residuals[b], B=B, seed=seed)
        rows.append([a, b, d, p])
    # Holm-Bonferroni adjustment across all pairs (descriptive)
    order = np.argsort([r[3] for r in rows])
    m = len(rows); running = 0.0
    adj = [None] * m
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i][3]))
        adj[i] = running
    for r, pa in zip(rows, adj):
        r.append(pa)
    return rows


def paired_realisation_test(x, y, B=5000, seed=0):
    """Synthetic benchmark: x, y are per-realisation RMSEs of two methods on
    the SAME realisations.  Returns (mean diff, bootstrap 68% CI, two-sided
    bootstrap p, Wilcoxon signed-rank p, fraction of realisations with x<y)."""
    from scipy.stats import wilcoxon
    x, y = np.asarray(x), np.asarray(y)
    m = ~np.isnan(x) & ~np.isnan(y)
    d = x[m] - y[m]
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    obs = d.mean()
    p = 2 * np.mean(boots <= 0) if obs >= 0 else 2 * np.mean(boots >= 0)
    try:
        pw = float(wilcoxon(d).pvalue)
    except ValueError:
        pw = np.nan
    return float(obs), float(np.percentile(boots, 16)), float(np.percentile(boots, 84)), \
        float(min(p, 1.0)), pw, float(np.mean(d < 0))
