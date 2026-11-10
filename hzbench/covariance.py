"""
Approximate CC systematic covariance (Moresco et al. 2020) and GLS fits.

Cov = diag(sigma_i^2) + eta(z_i) Hbar_i eta(z_j) Hbar_j, with eta(z) linearly
interpolated between 13.2% (z=0.07) and 3.9% (z=1.965).  This is an
APPROXIMATION to the full per-source budget of Moresco+2020 (their published
covariance is not used here), and sigma_i already contain each paper's own
systematic error, so the systematic part is partly double counted; the
analysis is a sensitivity test only.

Hbar is either the measured H_i (v1 choice, "measured") or the flat-LCDM
best fit to the training data ("model"), the latter removing any dependence
of the covariance on the measured values.  In each LOOCV fold the
covariance of the training points is the sub-matrix with the held-out row
and column removed, so the held-out measurement never enters the fit; the
held-out residual is standardised by sqrt(Cov_ii).

GLS methods: polynomial (Cholesky-whitened least squares), GP (full Cov in
the Gram matrix, amplitude and length scale by maximising the marginal
likelihood with Cov -- in v1 these were fixed, which is corrected here), NN
(loss r^T Cov^-1 r / N with the SAME optimiser settings as the baseline NN;
v1 used lr=1e-2 and 4000 epochs here).  Spline and nodal estimators take
diagonal weights only, sigma_i -> sqrt(Cov_ii).
"""
import numpy as np
from scipy.linalg import cholesky, cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize


def eta_model(z, z_lo=0.07, z_hi=1.965, eta_lo=0.132, eta_hi=0.039):
    z = np.atleast_1d(z).astype(float)
    return eta_lo + np.clip((z - z_lo) / (z_hi - z_lo), 0, 1) * (eta_hi - eta_lo)


def build_covariance(z, Hbar, sigma_stat):
    v = eta_model(z) * Hbar
    return np.diag(sigma_stat ** 2) + np.outer(v, v)


def fit_poly_gls(z, H, Cov, degree=4):
    L = cholesky(Cov, lower=True)
    X = np.vander(z, degree + 1, increasing=True)
    c, *_ = np.linalg.lstsq(solve_triangular(L, X, lower=True), solve_triangular(L, H, lower=True), rcond=None)
    poly = np.poly1d(c[::-1])
    return lambda zq: poly(np.atleast_1d(zq))


def _matern52(r, amp, ell):
    x = np.sqrt(5.0) * np.abs(r) / ell
    return amp * (1 + x + x ** 2 / 3.0) * np.exp(-x)


def fit_gp_gls(z, H, Cov):
    y = H - H.mean()
    R = z[:, None] - z[None, :]

    def nll(t):
        amp, ell = np.exp(t)
        K = _matern52(R, amp, ell) + Cov
        try:
            c = cho_factor(K)
        except np.linalg.LinAlgError:
            return 1e10
        return 0.5 * y @ cho_solve(c, y) + np.sum(np.log(np.diag(c[0])))

    best = None
    for t0 in ([np.log(np.var(y) + 1e-6), np.log(0.5)], [np.log(np.var(y) * 10 + 1e-6), np.log(2.0)],
               [np.log(np.var(y) + 1e-6), np.log(0.1)]):
        r = minimize(nll, t0, method="L-BFGS-B", bounds=[(-9, 18), (np.log(1e-3), np.log(1e3))])
        if best is None or r.fun < best.fun:
            best = r
    amp, ell = np.exp(best.x)
    alpha = cho_solve(cho_factor(_matern52(R, amp, ell) + Cov), y)
    return lambda zq: _matern52(np.atleast_1d(zq)[:, None] - z[None, :], amp, ell) @ alpha + H.mean()


def fit_nn_gls(z, H, Cov, epochs=2000, lr=1e-3, hidden=64, weight_decay=1e-4, init_seed=42):
    import torch, torch.nn as nn
    torch.manual_seed(init_seed)
    zm, zs, Hm, Hs = z.mean(), z.std(), H.mean(), H.std()
    zt = torch.tensor(((z - zm) / zs).reshape(-1, 1), dtype=torch.float32)
    Ht = torch.tensor(((H - Hm) / Hs).reshape(-1, 1), dtype=torch.float32)
    P = np.linalg.inv(Cov) * Hs ** 2
    P = P / np.mean(np.diag(P))          # same normalisation as the diagonal weights
    Pt = torch.tensor(P, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(z)
    for _ in range(epochs):
        opt.zero_grad()
        r = (model(zt) - Ht).flatten()
        (r @ Pt @ r / n).backward()
        opt.step()

    def predict(zq):
        zq = np.atleast_1d(zq).astype(float)
        with torch.no_grad():
            out = model(torch.tensor(((zq - zm) / zs).reshape(-1, 1), dtype=torch.float32)).numpy().flatten()
        return out * Hs + Hm
    return predict
