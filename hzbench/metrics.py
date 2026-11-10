"""
Performance metrics and derived diagnostics.

* integrated RMSE / MAE of a reconstruction against a known H(z) on a grid;
* deceleration parameter q(z) = (1+z) H'/H - 1 by central differences with
  step h (step-size sensitivity is examined in run_tier1.py);
* transition redshift z_t (first sign change of q);
* the Om(z) diagnostic of Sahni, Shafieloo & Starobinsky (2008),
  Om(z) = [E^2(z) - 1] / [(1+z)^3 - 1],  E = H(z)/H(0),
  which needs no derivative (H(0) is the reconstruction's own z=0 value);
* the parametric reference fits.  For flat LCDM truths the 2-parameter
  flat-LCDM fit is correctly specified.  For the non-flat and CPL truths of
  Tier 2 we also fit the correctly specified 3-parameter (H0, Om, Ok) and
  4-parameter (H0, Om, w0, wa) families; the flat-LCDM fit is then
  MISSPECIFIED and is reported only as a common anchor.

We do not compare against sigma/sqrt(N): that is the bound for estimating a
single constant from N independent points, not for a whole curve.
"""
import numpy as np
from scipy.optimize import curve_fit


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def integrated_rmse(predict_fn, cosmology, zgrid):
    pred = predict_fn(zgrid)
    true = cosmology.Hz(zgrid).flatten()
    return rmse(pred, true), mae(pred, true)


def deceleration_parameter(predict_fn, zgrid, h=1e-3):
    Hz = predict_fn(zgrid)
    dHdz = (predict_fn(zgrid + h) - predict_fn(zgrid - h)) / (2 * h)
    return (1 + zgrid) * dHdz / Hz - 1


def transition_redshift_from_curve(zgrid, qvals):
    sc = np.where(np.diff(np.sign(qvals)) != 0)[0]
    if len(sc) == 0:
        return np.nan
    i = sc[0]
    z1, z2, q1, q2 = zgrid[i], zgrid[i + 1], qvals[i], qvals[i + 1]
    return z1 - q1 * (z2 - z1) / (q2 - q1)


def om_diagnostic(predict_fn, zgrid):
    H0 = float(np.atleast_1d(predict_fn(np.array([0.0])))[0])
    E2 = (predict_fn(zgrid) / H0) ** 2
    return (E2 - 1.0) / ((1 + zgrid) ** 3 - 1.0)


# ─── parametric reference fits ──────────────────────────────────────────────
def flat_lcdm_model(z, H0, Om):
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def nonflat_lcdm_model(z, H0, Om, Ok):
    return H0 * np.sqrt(np.clip(Om * (1 + z) ** 3 + Ok * (1 + z) ** 2 + (1 - Om - Ok), 1e-8, None))


def cpl_model(z, H0, Om, w0, wa):
    a = 1.0 / (1 + z)
    de = a ** (-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))
    return H0 * np.sqrt(np.clip(Om * (1 + z) ** 3 + (1 - Om) * de, 1e-8, None))


PARAM_MODELS = {
    "flat_lcdm": (flat_lcdm_model, [70, 0.3], ([40, 0.05], [100, 0.95])),
    "nonflat_lcdm": (nonflat_lcdm_model, [70, 0.3, 0.0], ([40, 0.05, -0.5], [100, 0.95, 0.5])),
    "cpl_de": (cpl_model, [70, 0.3, -1.0, 0.0], ([40, 0.05, -2.5, -3.0], [100, 0.95, 0.5, 3.0])),
}


def parametric_fit(z, H, sigma, family="flat_lcdm"):
    model, p0, bounds = PARAM_MODELS[family]
    popt, pcov = curve_fit(model, z, H, p0=p0, sigma=sigma, absolute_sigma=True,
                           bounds=bounds, maxfev=20000)
    return (lambda zq: model(np.atleast_1d(zq), *popt)), popt, pcov


def parametric_reference_rmse(z, H_obs, sigma, cosmology, zgrid, family="flat_lcdm"):
    try:
        pred_fn, popt, _ = parametric_fit(z, H_obs, sigma, family)
    except Exception:
        return np.nan, None
    return rmse(pred_fn(zgrid), cosmology.Hz(zgrid).flatten()), popt
