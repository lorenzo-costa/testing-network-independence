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
#TODO this tests against full independence (i.e. full product copula) but i
# i don't really care about independence within Z or X (i.e. columns may be dependent)
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
# Block CvM statistic — inner kernels
# ---------------------------------------------------------------------------

def _build_M_numpy(V):
    """
    Helper for NumPy: Builds the (n, n) kernel matrix where 
    M_ij = Π_d min(V[i,d], V[j,d]). Memory footprint: O(n²).
    """
    n, d = V.shape
    V32 = V.astype(np.float32, copy=False)
    prod = np.ones((n, n), dtype=np.float32)
    for dd in range(d):
        vd = V32[:, dd]
        np.multiply(prod, np.minimum(vd[:, None], vd[None, :]), out=prod)
    return prod

def _cvm_block_terms_numpy(Vx, Vz):
    """
    NumPy fallback for block independence terms.
    Builds the Mx and Mz matrices and calculates the row sums.
    """
    Mx = _build_M_numpy(Vx)
    Mz = _build_M_numpy(Vz)
    
    term1_sum = np.sum(Mx * Mz)
    sum_Mx = Mx.sum(axis=1).astype(np.float64)
    sum_Mz = Mz.sum(axis=1).astype(np.float64)
    
    return float(term1_sum), sum_Mx, sum_Mz

if _HAS_NUMBA:
    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _cvm_block_terms_numba(Vx, Vz):
        """
        Numba parallel kernel for block independence.
        Calculates all required integrals in a single pass without 
        storing the full (n, n) kernel matrices. Memory footprint: O(n).
        """
        n, dx = Vx.shape
        _, dz = Vz.shape

        term1_sum = 0.0
        sum_Mx = np.zeros(n, dtype=nb.float64)
        sum_Mz = np.zeros(n, dtype=nb.float64)

        for i in nb.prange(n):          # parallel over rows
            row_term1 = 0.0
            row_Mx = 0.0
            row_Mz = 0.0
            
            for j in range(n):
                # Compute kernel for X: Π min(Vx[i], Vx[j])
                mx_ij = 1.0
                for d in range(dx):
                    vxi = Vx[i, d]
                    vxj = Vx[j, d]
                    mx_ij *= vxi if vxi < vxj else vxj
                    
                # Compute kernel for Z: Π min(Vz[i], Vz[j])
                mz_ij = 1.0
                for d in range(dz):
                    vzi = Vz[i, d]
                    vzj = Vz[j, d]
                    mz_ij *= vzi if vzi < vzj else vzj
                
                row_term1 += mx_ij * mz_ij
                row_Mx += mx_ij
                row_Mz += mz_ij
                
            # Store row sums
            sum_Mx[i] = row_Mx
            sum_Mz[i] = row_Mz
            term1_sum += row_term1
            
        return term1_sum, sum_Mx, sum_Mz

    _cvm_block_terms = _cvm_block_terms_numba
else:
    _cvm_block_terms = _cvm_block_terms_numpy


# ---------------------------------------------------------------------------
# CvM statistic — public interface
# ---------------------------------------------------------------------------

def cvm_stat_block_independence(X, Z):
    """
    Cramér–von Mises statistic for block independence between vectors X and Z.
    
    Tests H_0: C_XZ(u, v) = C_X(u) C_Z(v). 
    This allows internal dimensions of X to be dependent on each other, and 
    internal dimensions of Z to be dependent on each other.

    Parameters
    ----------
    X : (n, d_x) array of latent positions
    Z : (n, d_z) array of latent positions

    Returns
    -------
    float
    """
    n = X.shape[0]
    
    # 1. Transform to empirical pseudo-observations
    Ux = pseudo_obs(X)
    Uz = pseudo_obs(Z)

    # 2. V = 1 - U (used in the min-product integral identity)
    Vx = (1.0 - Ux).astype(np.float64)
    Vz = (1.0 - Uz).astype(np.float64)

    # 3. Compute kernel sums efficiently
    term1_sum, sum_Mx, sum_Mz = _cvm_block_terms(Vx, Vz)

    # 4. Reconstruct the three integral terms
    # Term 1: ∫ (C_nXZ)²
    term1 = term1_sum / (n ** 2)
    
    # Term 2: ∫ C_nXZ * C_nX * C_nZ
    term2 = np.sum(sum_Mx * sum_Mz) / (n ** 3)
    
    # Term 3: ∫ (C_nX)² * (C_nZ)²
    term3 = (np.sum(sum_Mx) * np.sum(sum_Mz)) / (n ** 4)

    # 5. Final CvM statistic
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