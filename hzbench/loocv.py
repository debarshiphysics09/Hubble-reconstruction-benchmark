"""Leave-one-out cross-validation with fold-by-fold output."""
import numpy as np
from hzbench.metrics import parametric_fit


def loocv_residuals(fitter, z, H, sigma):
    """fitter(z, H, sigma) -> predict.  Returns (chi, standardised residuals,
    predictions).  Residual convention: r_i = (Hhat_{-i}(z_i) - H_i)/sigma_i."""
    n = len(z)
    pred = np.full(n, np.nan)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        try:
            pred[i] = float(np.atleast_1d(fitter(z[m], H[m], sigma[m])(np.array([z[i]])))[0])
        except Exception:
            pred[i] = np.nan
    r = (pred - H) / sigma
    return float(np.sqrt(np.nanmean(r ** 2))), r, pred


def parametric_fitter(family="flat_lcdm"):
    return lambda z, H, s: parametric_fit(z, H, s, family)[0]
