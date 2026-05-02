import numpy as np
from numpy.linalg import eigh, solve

# ====== core: fit params on TRAIN biomarkers only ======
def fit_gaussian_knockoff_params(Xb_tr, eps=1e-12, cap_s=0.99, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    n, p = Xb_tr.shape
    mu = Xb_tr.mean(axis=0)
    Xc = Xb_tr - mu
    Sigma = (Xc.T @ Xc) / (n - 1)

    # Equi-correlated S = s I, with s in (0, 2*lambda_min]
    w, V = eigh(Sigma)
    lam_min = max(float(w.min()), eps)
    s = min(2.0 * lam_min, cap_s)  # cap for numerical stability when standardized
    S = np.eye(p) * s

    # A = I - Sigma^{-1} S  (solve for stability)
    Sigma_inv_S = solve(Sigma, S)
    A = np.eye(p) - Sigma_inv_S

    # middle = 2S - S Sigma^{-1} S, then C s.t. C^T C = middle (PSD)
    middle = 2 * S - S @ Sigma_inv_S
    # Force PSD via spectral clip (handles tiny negatives from roundoff)
    wm, Vm = eigh((middle + middle.T) / 2.0)
    wm = np.clip(wm, a_min=eps, a_max=None)
    C = (Vm * np.sqrt(wm)) @ Vm.T

    return {"mu": mu, "Sigma": Sigma, "S": S, "A": A, "C": C, "rng": rng}

# ====== sample knockoffs for ANY matrix (train or test) using fitted params ======
def sample_knockoffs(Xb, params):
    mu, A, C, rng = params["mu"], params["A"], params["C"], params["rng"]
    Z = rng.normal(size=Xb.shape)
    return (Xb - mu) @ A + Z @ C + mu

# ====== convenience wrapper for your TRAIN matrix [biomarkers | confounders] ======
def make_train_knockoffs(X_tr, seed=42):
    rng = np.random.default_rng(seed)
    ko_params = fit_gaussian_knockoff_params(X_tr, rng=rng)
    Xb_tilde_tr = sample_knockoffs(X_tr, ko_params)
    return Xb_tilde_tr, ko_params


def make_test_knockoffs(X_te, ko_params):
    Xb_tilde_te = sample_knockoffs(X_te, ko_params)
    return Xb_tilde_te
