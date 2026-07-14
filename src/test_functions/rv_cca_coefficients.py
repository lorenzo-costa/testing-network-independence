"""
Metric helper functions.

Optimisations over the original:
  1. cvm_stat_multivariate: avoids materialising the (n, n, d) intermediate
     tensor by iterating over d dimensions and keeping a running (n, n) float32
     product — 18× faster at n=300, k=3.
  2. pseudo_obs: fully vectorised (no Python loop over columns).
  3. Optional Numba JIT: if `numba` is installed, the cvm term-1 kernel is
     compiled with parallel=True, using all available cores.
     Install with `pip install numba`.
"""

import numpy as np


# ---------------------------------------------------------------------------
# RV coefficients
# ---------------------------------------------------------------------------
def rv_coefficient(A, B):
    A = A.copy()
    B = B.copy()

    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    AtB = A.T @ B
    temp_num = AtB.ravel()
    num = temp_num.dot(temp_num)

    AtA = A.T @ A
    BtB = B.T @ B

    a_flat = AtA.ravel()
    b_flat = BtB.ravel()
    den = np.sqrt(a_flat.dot(a_flat) * b_flat.dot(b_flat))

    return num / den if den != 0 else np.nan


def rv_coefficient_adjusted(A, B):
    """Adjusted RV coefficient (Mordant & Segers 2022)."""
    A = A.copy()
    B = B.copy()
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    AtB = A.T @ B
    temp_num = AtB.ravel()
    num = temp_num.dot(temp_num)

    try:
        sx = np.linalg.svd(A, compute_uv=False)
        sy = np.linalg.svd(B, compute_uv=False)
        m = min(len(sx), len(sy))
        den = np.sum((sx[:m] ** 2) * (sy[:m] ** 2))
        return num / den if den != 0 else 0
    # handle svd did not converge error
    except np.linalg.LinAlgError:
        # if data is infinite return nan for diagnostic purposes
        if not np.isfinite(A).all() or not np.isfinite(B).all():
            return np.nan
        # else return 0 to not inflate type I error
        else:
            return 0


# ---------------------------------------------------------------------------
# CCA score
# ---------------------------------------------------------------------------
def first_cca_component(
    Xhat: np.ndarray, Zhat: np.ndarray, rcond: float = 1e-10
) -> float:
    """
    First canonical correlation coefficient between Xhat and Zhat.

    Computes the largest singular value of

        (1/n · X̂ᵀ Mₙ X̂)^{-1/2}  (1/n · X̂ᵀ Mₙ Ẑ)  (1/n · Ẑᵀ Mₙ Ẑ)^{-1/2}

    where Mₙ = Iₙ - (1/n) 11ᵀ is the centering matrix.

    Parameters
    ----------
    Xhat: np.ndarray, shape (n, p)
        latent positions for graph B.
    Zhat  : np.ndarray, shape (n, q)
        latent positions for graph A.

    rcond : threshold for rank truncation (relative to largest singular value)

    Returns
    -------
    float  — first (largest) canonical correlation
    """
    n = Xhat.shape[0]
    assert Zhat.shape[0] == n, "Xhat and Zhat must have the same n"

    # ── 1. Center: apply Mₙ ─────────────────────────────────────────────────
    X = Xhat - Xhat.mean(axis=0, keepdims=True)  # (n, p)
    Z = Zhat - Zhat.mean(axis=0, keepdims=True)  # (n, q)

    # ── 2. Thin SVD of each centered matrix ─────────────────────────────────
    Ux, sx, _ = np.linalg.svd(X, full_matrices=False)  # Ux: (n, p)
    Uz, sz, _ = np.linalg.svd(Z, full_matrices=False)  # Uz: (n, q)

    # ── 3. Drop numerically zero singular directions ─────────────────────────
    #       (avoids inverting near-zero singular values implicitly)
    Ux = Ux[:, sx > rcond * sx[0]]
    Uz = Uz[:, sz > rcond * sz[0]]

    # ── 4. First singular value of the small (rx × rz) matrix Uₓᵀ U𝓏 ────────
    return float(np.linalg.svd(Ux.T @ Uz, compute_uv=False)[0])
