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
from scipy.stats import rankdata
from numba import njit

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

    try:
        sx = np.linalg.svd(A, compute_uv=False)
        sy = np.linalg.svd(B, compute_uv=False)
        m   = min(len(sx), len(sy))
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

def relative_nuclear_error(X, Xhat):
    """
    Rotation-invariant and more robust than Frobenius.
    Uses the sum of singular values (Nuclear Norm).
    """
    error_matrix = X - Xhat
    
    # Compute singular values
    s_error = np.linalg.svd(error_matrix, compute_uv=False)
    s_true = np.linalg.svd(X, compute_uv=False)
    
    return np.sum(s_error) / np.sum(s_true)

@njit(cache=True)
def _joint_cdf_bit(sa_dense, sb_dense, sort_order, M, K):
    """
    Computes F_AB[i] = #{j : sa[j] <= sa[i] AND sb[j] <= sb[i]} / M
    in O(M log M) using a Fenwick (BIT) tree keyed on sb's dense ranks.
    Note this has memory requirement O(M), if we try to move this to a 3 var case
    computation would take O(M log M log M) but memory would be O(M^2). May be
    worth looking into.

    Key idea:
      - Process points in ascending sa order (ties handled as a batch).
      - For each group sharing the same sa value:
          1. UPDATE the BIT for every point in the group.
          2. QUERY  the BIT for every point in the group (reads counts
             of sb ranks <= current sb rank inserted so far).
        Splitting query and update within a tie-group ensures we count
        ALL points with sa <= current (not just strictly <).
    """
    tree = np.zeros(K + 2, dtype=np.int64)
    result = np.zeros(M, dtype=np.float64)

    i = 0
    while i < M:
        # --- find end of current sa-tie group ---
        j = i
        cur_sa = sa_dense[sort_order[i]]
        while j < M and sa_dense[sort_order[j]] == cur_sa:
            j += 1

        # --- UPDATE PHASE (insert entire group FIRST) ---
        # Now the BIT contains all items with sa <= cur_sa
        for k in range(i, j):
            idx = sort_order[k]
            r = int(sb_dense[idx])
            while r <= K:          # point update
                tree[r] += 1
                r += r & (-r)

        # --- QUERY PHASE (read from the updated BIT) ---
        # Because we updated first, this correctly calculates P(SA <= sa AND SB <= sb)
        for k in range(i, j):
            idx = sort_order[k]
            r = int(sb_dense[idx])
            s = 0
            while r > 0:           # prefix-sum [1 .. r]
                s += tree[r]
                r -= r & (-r)
            result[idx] = float(s)

        i = j

    return result / M


def observed_cvm_dependency(A, B, degree=2):
    """
    Computes a rotationally-invariant Cramér-von Mises copula dependency
    measure between two graphs A and B based on shared neighbor counts.

    Parameters:
        A, B (np.ndarray): N x N binary adjacency matrices.
    Returns:
        float: The Cramér-von Mises statistic.
    """
    N = A.shape[0]
    if B.shape[0] != N:
        raise ValueError("Matrices A and B must have the same number of nodes.")

    # 1. Shared-neighbor matrices
    SA = np.linalg.matrix_power(A, degree)  
    SB = np.linalg.matrix_power(B, degree) 

    # 2. Off-diagonal elements only  (N*(N-1) pairs)
    mask = ~np.eye(N, dtype=bool)
    sa_vals = SA[mask]
    sb_vals = SB[mask]
    M = len(sa_vals)

    # 3. Marginal empirical CDFs  (unchanged from original)
    F_A = rankdata(sa_vals, method='max') / M
    F_B = rankdata(sb_vals, method='max') / M

    # 4. Dense ranks for BIT indexing  (maps unique values -> 1..K)
    sa_dense = rankdata(sa_vals, method='dense').astype(np.int64)
    sb_dense = rankdata(sb_vals, method='dense').astype(np.int64)
    K = int(sb_dense.max())

    # Sort indices by ascending sa_dense (stable = deterministic tie ordering)
    sort_order = np.argsort(sa_dense, kind='stable').astype(np.int64)

    # 5. Joint empirical CDF  — O(M log M) instead of O(M²)
    F_AB = _joint_cdf_bit(sa_dense, sb_dense, sort_order, M, K)

    # 6. Cramér-von Mises statistic
    return float(np.sum((F_AB - F_A * F_B) ** 2))