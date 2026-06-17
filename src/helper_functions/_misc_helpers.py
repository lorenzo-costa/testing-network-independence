import numpy as np
from scipy.linalg import blas, norm
from numba import njit

# ---------------------------------------------------------------------------
# error functions
# ---------------------------------------------------------------------------
def mse(X, Xhat):
    return ((X - Xhat) ** 2).mean()


def relative_frobenius_norm(X, Xhat, inplace=True):
    if not inplace:
        den = norm(X, "fro")
        return 0 if den == 0 else norm(Xhat - X, "fro") / den

    X_flat = X.ravel()
    Xhat_flat = Xhat.ravel()
    den = blas.dnrm2(X_flat)
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

# ---------------------------------------------------------------------------
# other helpers
# ---------------------------------------------------------------------------

# needed for graph correlation
def u_center(K: np.ndarray) -> np.ndarray:
    """
    Apply U-centering to an n×n kernel matrix.
    
    For i ≠ j:
        K̃_ij = K_ij - 1/(n-2) * ΣᵥK_iv - 1/(n-2) * ΣᵤK_uj + 1/((n-1)(n-2)) * Σ_{u,v}K_uv
    Diagonal entries are set to 0.
    """
    n = K.shape[0]
    row_sums = K.sum(axis=1, keepdims=True)   # (n, 1)
    col_sums = K.sum(axis=0, keepdims=True)   # (1, n)
    total    = K.sum()                         # scalar

    K_tilde = (K
               - row_sums / (n - 2)
               - col_sums / (n - 2)
               + total   / ((n - 1) * (n - 2)))
    np.fill_diagonal(K_tilde, 0.0)
    return K_tilde


# needed for speeding up CvM computation
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
            while r <= K:  # point update
                tree[r] += 1
                r += r & (-r)

        # --- QUERY PHASE (read from the updated BIT) ---
        # Because we updated first, this correctly calculates P(SA <= sa AND SB <= sb)
        for k in range(i, j):
            idx = sort_order[k]
            r = int(sb_dense[idx])
            s = 0
            while r > 0:  # prefix-sum [1 .. r]
                s += tree[r]
                r -= r & (-r)
            result[idx] = float(s)

        i = j

    return result / M

def _as_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x