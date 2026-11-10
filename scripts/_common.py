"""Shared set-up for the driver scripts (paths, method dictionary)."""
import os, sys, warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.makedirs("results", exist_ok=True)
warnings.filterwarnings("ignore")
import numpy as np
from hzbench import methods as M

FITTERS = {
    "GP": lambda z, H, s: M.fit_gp(z, H, s),
    "Polynomial": lambda z, H, s: M.fit_poly(z, H, s),
    "NN": lambda z, H, s: M.fit_nn(z, H, s),
    "Spline": lambda z, H, s: M.fit_spline(z, H, s),
    "Nodal": lambda z, H, s: M.fit_nodal(z, H, s),           # v1 prescription, bins on [0, 2]
}
ORDER = M.METHOD_ORDER
FAST = os.environ.get("HZ_FAST", "0") == "1"    # quick smoke-test mode


def log(*a):
    print(*a, flush=True)
