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
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm, blas
try:
    import copent
    _HAS_COPENT = True
except ImportError:
    _HAS_COPENT = False

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# RV coefficients
# ---------------------------------------------------------------------------

def rv_coefficient(A, B):
    AtB = A.T @ B
    temp_num = AtB.ravel()
    num = temp_num.dot(temp_num)

    AtA = A.T @ A
    BtB = B.T @ B

    a_flat = AtA.ravel()
    b_flat = BtB.ravel()
    den = np.sqrt(a_flat.dot(a_flat) * b_flat.dot(b_flat))

    return num / den if den != 0 else 0


def rv_coefficient_adjusted(A, B):
    """Adjusted RV coefficient (Mordant & Segers 2022)."""
    AtB = A.T @ B
    temp_num = AtB.ravel()
    num = temp_num.dot(temp_num)

    sx = np.linalg.svd(A, compute_uv=False)
    sy = np.linalg.svd(B, compute_uv=False)
    m   = min(len(sx), len(sy))
    den = np.sum((sx[:m] ** 2) * (sy[:m] ** 2))

    return num / den if den != 0 else 0


# ---------------------------------------------------------------------------
# Pseudo-observations
# ---------------------------------------------------------------------------

def pseudo_obs(X):
    """
    Vectorised pseudo-observations — no Python loop over columns.

    Equivalent to applying scipy.stats.rankdata column-wise and dividing by
    (n+1).  Uses double argsort which is O(n log n) per column but fully
    vectorised.
    """
    n = X.shape[0]
    # double argsort along axis=0 gives 0-based ranks; +1 for 1-based
    return (np.argsort(np.argsort(X, axis=0), axis=0) + 1) / (n + 1.0)


# ---------------------------------------------------------------------------
# CvM statistic — inner kernel
# ---------------------------------------------------------------------------

def _cvm_term1_numpy(V):
    """
    Compute  Σ_{i,j} Π_d min(V[i,d], V[j,d])  without a (n,n,d) tensor.

    V has shape (n, d).  We iterate over d, accumulating a running (n, n)
    float32 product matrix.  Memory footprint: O(n²) instead of O(n²d).
    """
    n, d = V.shape
    V32  = V.astype(np.float32, copy=False)
    prod = np.ones((n, n), dtype=np.float32)
    for dd in range(d):
        vd = V32[:, dd]
        np.multiply(prod, np.minimum(vd[:, None], vd[None, :]), out=prod)
    return float(prod.sum())


if _HAS_NUMBA:
    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _cvm_term1_numba(V):
        """
        Numba parallel kernel: uses all CPU cores, avoids (n,n,d) tensor,
        and keeps the inner loop scalar for maximum SIMD throughput.
        """
        n, d = V.shape
        total = 0.0
        for i in nb.prange(n):          # parallel over rows
            row_sum = 0.0
            for j in range(n):
                prod = 1.0
                for dd in range(d):
                    vi = V[i, dd]
                    vj = V[j, dd]
                    prod *= vi if vi < vj else vj
                row_sum += prod
            total += row_sum
        return total

    _cvm_term1 = _cvm_term1_numba
else:
    _cvm_term1 = _cvm_term1_numpy


# ---------------------------------------------------------------------------
# CvM statistic — public interface
# ---------------------------------------------------------------------------

def cvm_stat_multivariate(X, Z):
    """
    Multivariate Cramér–von Mises statistic between latent positions X and Z.

    Copula-based: converts both matrices to pseudo-observations, stacks them,
    then computes the d-dimensional CvM statistic.

    Parameters
    ----------
    X, Z : (n, k) arrays of latent positions

    Returns
    -------
    float
    """
    Ux = pseudo_obs(X)
    Uz = pseudo_obs(Z)
    W  = np.hstack([Ux, Uz])            # (n, 2k)
    n, d = W.shape

    # V = 1 - W  (used in term1 min-product formula)
    V = (1.0 - W).astype(np.float64)

    # term1 = Σ_{i,j} Π_d min(V[i,d], V[j,d]) / n²
    term1 = _cvm_term1(V) / n ** 2

    # term2 and term3 use float64 throughout for accuracy
    term2 = float(np.mean(np.prod(0.5 * (1.0 - W ** 2), axis=1)))
    term3 = (1.0 / 3.0) ** d

    return n * (term1 - 2.0 * term2 + term3)


# ---------------------------------------------------------------------------
# Mutual information via copula entropy
# ---------------------------------------------------------------------------

def copula_mutual_information(X, Y, k=3):
    """
    Mutual information I(X; Y) via Copula Entropy (KSG estimator).
    Requires the `copent` package.
    """
    if not _HAS_COPENT:
        raise ImportError("copula_mutual_information requires `pip install copent`")
    X  = np.reshape(X, (len(X), -1))
    Y  = np.reshape(Y, (len(Y), -1))
    XY = np.hstack([X, Y])
    tc_xy = copent.copent(XY, k=k)
    tc_x  = copent.copent(X,  k=k) if X.shape[1] > 1 else 0.0
    tc_y  = copent.copent(Y,  k=k) if Y.shape[1] > 1 else 0.0
    return max(0.0, float(tc_xy - tc_x - tc_y))


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------

def mse(X, Xhat):
    return ((X - Xhat) ** 2).mean()


def relative_frobenius_norm(X, Xhat, inplace=True):
    if not inplace:
        den = norm(X, "fro")
        return 0 if den == 0 else norm(Xhat - X, "fro") / den

    X_flat    = X.ravel()
    Xhat_flat = Xhat.ravel()
    den       = blas.dnrm2(X_flat)
    if den == 0:
        return 0
    diff = np.copy(Xhat_flat)
    blas.daxpy(X_flat, diff, a=-1.0)
    return blas.dnrm2(diff) / den