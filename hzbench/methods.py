"""
Reconstruction methods benchmarked in this work.

Every fitter has the signature  fit_xxx(z, H, sigma, **options) -> predict(zq)
and returns a callable that maps redshifts to reconstructed H(z).

Baseline configurations (the ones used for the main benchmark tables)
---------------------------------------------------------------------
  NN      1-64-64-1 tanh network (4,353 trainable parameters), inverse-variance
          weighted MSE, Adam (lr=1e-3), weight decay 1e-4, 2000 full-batch epochs,
          weight-initialisation seed 42 unless `init_seed` is given.
  Poly    weighted least-squares polynomial, degree 4.
  Spline  scipy UnivariateSpline, cubic, weights 1/sigma, FIXED smoothing
          factor s = N (a residual-target heuristic; NOT generalised
          cross-validation).
  GP      Matern-5/2 kernel; amplitude and length scale by maximising the log
          marginal likelihood (5 optimiser restarts); per-point sigma_i^2 on
          the diagonal (no extra fitted white-noise term).
  Nodal   K equal-width bins; inverse-variance weighted mean per bin, placed
          at the bin centre; linear interpolation between populated nodes;
          CONSTANT extrapolation outside the outermost populated nodes.
          The original (v1) prescription fixes the bin range to [0, 2], the
          synthetic-benchmark domain.  Data outside [0, 2] then fall in no bin
          and are predicted by constant extrapolation; this is what produced
          the BAO-only failure reported in v1 (see scripts/run_nodal_diagnostics.py).

Variants added in revision (all optional keyword arguments)
-----------------------------------------------------------
  fit_nodal(..., zmin, zmax)               bin range (e.g. the data range)
  fit_nodal(..., n_bins)                   number of bins
  fit_nodal(..., binning="quantile")       equal-count (density-adaptive) bins,
                                           nodes at the weighted mean redshift
  fit_nn(..., init_seed, weight_decay)     initialisation seed / regularisation
  fit_poly(..., degree), fit_spline(..., s_factor), fit_gp(..., length_scale)
                                           used by the common-tuning experiment
"""
import numpy as np
try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    nn = None
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

NN_INIT_SEED = 42      # default weight-initialisation seed (independent of the data seed)
NN_HIDDEN = 64
NN_EPOCHS = 2000
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4


def nn_parameter_count(hidden=NN_HIDDEN):
    """1-h-h-1 fully connected network with biases: (1h+h) + (h^2+h) + (h+1)."""
    return (1 * hidden + hidden) + (hidden * hidden + hidden) + (hidden * 1 + 1)


# ─── Neural network (PyTorch) ────────────────────────────────────────────────
if _HAS_TORCH:
    class HzNet(nn.Module):
        def __init__(self, hidden=NN_HIDDEN):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x)


def fit_nn(z, H, sigma, epochs=NN_EPOCHS, lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
           seed=0, hidden=NN_HIDDEN, init_seed=None, return_losses=False):
    """`seed` is accepted for API compatibility and ignored; the weight
    initialisation is controlled by `init_seed` (default NN_INIT_SEED=42)."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for fit_nn; use fit_nn_numpy instead")
    torch.manual_seed(NN_INIT_SEED if init_seed is None else int(init_seed))
    z_mean, z_std = z.mean(), z.std()
    H_mean, H_std = H.mean(), H.std()
    zt = torch.tensor(((z - z_mean) / z_std).reshape(-1, 1), dtype=torch.float32)
    Ht = torch.tensor(((H - H_mean) / H_std).reshape(-1, 1), dtype=torch.float32)
    w = torch.tensor((1.0 / sigma ** 2).reshape(-1, 1), dtype=torch.float32)
    w = w / w.mean()
    model = HzNet(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = (w * (model(zt) - Ht) ** 2).mean()
        loss.backward()
        opt.step()
        if return_losses:
            losses.append(float(loss.item()))

    def predict(zq):
        zq = np.atleast_1d(zq).astype(float)
        with torch.no_grad():
            zqt = torch.tensor(((zq - z_mean) / z_std).reshape(-1, 1), dtype=torch.float32)
            out = model(zqt).numpy().flatten()
        return out * H_std + H_mean

    return (predict, losses) if return_losses else predict


# ─── Polynomial (weighted least squares) ─────────────────────────────────────
def fit_poly(z, H, sigma, degree=4):
    coeffs = np.polyfit(z, H, degree, w=1.0 / sigma)
    poly = np.poly1d(coeffs)
    return lambda zq: poly(np.atleast_1d(zq))


# ─── Smoothing spline (fixed residual-target smoothing, s = s_factor * N) ───
def fit_spline(z, H, sigma, s_factor=1.0):
    order = np.argsort(z)
    zs, Hs, sig_s = z[order], H[order], sigma[order]
    zs_unique, idx = np.unique(zs, return_index=True)   # UnivariateSpline needs strictly increasing z
    Hs, sig_s = Hs[idx], sig_s[idx]
    n = len(zs_unique)
    spl = UnivariateSpline(zs_unique, Hs, w=1.0 / sig_s, k=min(3, n - 1), s=s_factor * n)
    return lambda zq: spl(np.atleast_1d(zq))


# ─── Gaussian process (Matern-5/2, marginal-likelihood hyperparameters) ─────
def fit_gp(z, H, sigma, length_scale=None, return_model=False):
    """length_scale=None: amplitude and length scale by ML-II (baseline).
    length_scale=x: length scale fixed at x, amplitude by ML-II (used only
    by the common-tuning experiment)."""
    X = z.reshape(-1, 1)
    Hmean = H.mean()
    if length_scale is None:
        kern = (ConstantKernel(np.var(H - Hmean) + 1e-6, (1e-4, 1e8))
                * Matern(length_scale=0.5, length_scale_bounds=(1e-3, 1e3), nu=2.5))
        restarts = 5
    else:
        kern = (ConstantKernel(np.var(H - Hmean) + 1e-6, (1e-4, 1e8))
                * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5))
        restarts = 0
    gp = GaussianProcessRegressor(kernel=kern, alpha=sigma ** 2, normalize_y=False,
                                  n_restarts_optimizer=restarts, random_state=0)
    gp.fit(X, H - Hmean)
    predict = lambda zq: gp.predict(np.atleast_1d(zq).reshape(-1, 1)) + Hmean
    if return_model:
        def predict_std(zq):
            m, s = gp.predict(np.atleast_1d(zq).reshape(-1, 1), return_std=True)
            return m + Hmean, s
        return predict, predict_std, gp
    return predict


# ─── Nodal / binned reconstruction ───────────────────────────────────────────
def fit_nodal(z, H, sigma, n_bins=8, zmin=0.0, zmax=2.0, binning="uniform",
              return_nodes=False):
    """binning="uniform": K equal-width bins on [zmin, zmax], node at the bin centre.
    binning="quantile": K bins holding (nearly) equal numbers of the supplied
    points, node at the inverse-variance-weighted mean redshift of the bin
    (zmin/zmax ignored).  Empty bins are skipped; outside the outermost
    populated nodes the curve is held constant (as in v1)."""
    order = np.argsort(z)
    z, H, sigma = z[order], H[order], sigma[order]
    centres, values = [], []
    if binning == "uniform":
        edges = np.linspace(zmin, zmax, n_bins + 1)
        for i in range(n_bins):
            hi_ok = (z < edges[i + 1]) if i < n_bins - 1 else (z <= edges[i + 1])
            mask = (z >= edges[i]) & hi_ok
            if mask.sum() == 0:
                continue
            w = 1.0 / sigma[mask] ** 2
            centres.append(0.5 * (edges[i] + edges[i + 1]))
            values.append(np.sum(w * H[mask]) / np.sum(w))
    elif binning == "quantile":
        k = int(min(n_bins, len(z)))
        for chunk in np.array_split(np.arange(len(z)), k):
            if len(chunk) == 0:
                continue
            w = 1.0 / sigma[chunk] ** 2
            centres.append(np.sum(w * z[chunk]) / np.sum(w))
            values.append(np.sum(w * H[chunk]) / np.sum(w))
    else:
        raise ValueError(binning)
    centres, values = np.array(centres), np.array(values)
    if len(centres) == 1:
        f = lambda zq: np.full(np.atleast_1d(zq).shape, values[0])
    else:
        f = interp1d(centres, values, kind="linear", bounds_error=False,
                     fill_value=(values[0], values[-1]))
    predict = lambda zq: np.asarray(f(np.atleast_1d(zq)), dtype=float)
    return (predict, centres, values) if return_nodes else predict


METHODS = {
    "NN": fit_nn,
    "Polynomial": fit_poly,
    "Spline": fit_spline,
    "GP": fit_gp,
    "Nodal": fit_nodal,
}
METHOD_ORDER = ["GP", "Polynomial", "NN", "Spline", "Nodal"]


# ─── NumPy re-implementation (independent cross-check of the PyTorch NN) ────
def fit_nn_numpy(z, H, sigma, epochs=NN_EPOCHS, lr=NN_LR, hidden=NN_HIDDEN, seed=0,
                 weight_decay=NN_WEIGHT_DECAY, init_seed=None, return_losses=False):
    """Pure-NumPy network with the same architecture, loss, optimiser and
    hyperparameters as fit_nn.  Its initialisation is drawn from a different
    generator (N(0,1) / N(0,1/h) weights, zero biases) and it applies the L2
    term to weights only, so it is an *implementation* cross-check, not a
    bit-for-bit replica.  Used only in the implementation-variability test."""
    rng = np.random.default_rng(NN_INIT_SEED if init_seed is None else int(init_seed))
    zc = (z - z.mean()) / z.std()
    Hc = (H - H.mean()) / H.std()
    w = (1.0 / sigma ** 2); w = w / w.mean()
    n, h = len(z), hidden
    W1 = rng.normal(0, 1.0, (1, h)); b1 = np.zeros(h)
    W2 = rng.normal(0, 1.0 / np.sqrt(h), (h, h)); b2 = np.zeros(h)
    W3 = rng.normal(0, 1.0 / np.sqrt(h), (h, 1)); b3 = np.zeros(1)
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    X, y, ww = zc.reshape(-1, 1), Hc.reshape(-1, 1), w.reshape(-1, 1)
    losses = []
    for t in range(1, epochs + 1):
        a1 = np.tanh(X @ W1 + b1); a2 = np.tanh(a1 @ W2 + b2); pred = a2 @ W3 + b3
        err = pred - y
        if return_losses:
            losses.append(float(np.mean(ww * err ** 2)))
        g = 2 * ww * err / n
        gW3 = a2.T @ g + 2 * weight_decay * W3; gb3 = g.sum(0)
        d2 = g @ W3.T * (1 - a2 ** 2)
        gW2 = a1.T @ d2 + 2 * weight_decay * W2; gb2 = d2.sum(0)
        d1 = d2 @ W2.T * (1 - a1 ** 2)
        gW1 = X.T @ d1 + 2 * weight_decay * W1; gb1 = d1.sum(0)
        for i, (p, gr) in enumerate(zip(params, [gW1, gb1, gW2, gb2, gW3, gb3])):
            m[i] = beta1 * m[i] + (1 - beta1) * gr
            v[i] = beta2 * v[i] + (1 - beta2) * gr ** 2
            p -= lr * (m[i] / (1 - beta1 ** t)) / (np.sqrt(v[i] / (1 - beta2 ** t)) + eps)

    def predict(zq):
        zq = np.atleast_1d(zq).astype(float)
        Xq = ((zq - z.mean()) / z.std()).reshape(-1, 1)
        a1 = np.tanh(Xq @ W1 + b1); a2 = np.tanh(a1 @ W2 + b2)
        return (a2 @ W3 + b3).flatten() * H.std() + H.mean()

    return (predict, losses) if return_losses else predict


def get_fitter(name, **opts):
    """Return f(z, H, sigma) for a method name, with fixed options bound."""
    base = METHODS[name]
    return lambda z, H, s: base(z, H, s, **opts)
