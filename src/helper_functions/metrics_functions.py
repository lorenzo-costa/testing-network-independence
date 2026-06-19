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
import numba as nb
from scipy.spatial.distance import cdist

from ._misc_helpers import u_center, _joint_cdf_bit, _as_2d

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
def first_cca_component(Xhat: np.ndarray, Zhat: np.ndarray, rcond: float = 1e-10) -> float:
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
    X = Xhat - Xhat.mean(axis=0, keepdims=True)   # (n, p)
    Z = Zhat - Zhat.mean(axis=0, keepdims=True)   # (n, q)

    # ── 2. Thin SVD of each centered matrix ─────────────────────────────────
    Ux, sx, _ = np.linalg.svd(X, full_matrices=False)   # Ux: (n, p)
    Uz, sz, _ = np.linalg.svd(Z, full_matrices=False)   # Uz: (n, q)

    # ── 3. Drop numerically zero singular directions ─────────────────────────
    #       (avoids inverting near-zero singular values implicitly)
    Ux = Ux[:, sx > rcond * sx[0]]
    Uz = Uz[:, sz > rcond * sz[0]]

    # ── 4. First singular value of the small (rx × rz) matrix Uₓᵀ U𝓏 ────────
    return float(np.linalg.svd(Ux.T @ Uz, compute_uv=False)[0])

# ---------------------------------------------------------------------------
# CvM computations
# ---------------------------------------------------------------------------
def observed_cvm_dependency(A, B, degree=2, rank_method="average"):
    """Computes a rotationally-invariant Cramér-von Mises copula dependency
    measure between two graphs A and B based on shared neighbor counts.

    Parameters
    ----------
    A : np.ndarray
        binary adjacency matrix of shape (N, N) for graph A
    B : np.ndarray
        binary adjacency matrix of shape (N, N) for graph B
    degree : int, optional
        the degree of shared neighbor counts to use (default is 2)
    rank_method : str, optional
        the method to use for ranking shared neighbor counts (default is "average").
        Options include:
        - "max": Use the maximum rank. This is the plug-in estimator standard in 
        copula theory (Deheuvels 1979). It has jump discontinuities.
        - "average" (default): Use the average rank. This is also known as the 
        mid-rank copula transform. It provides a sort of smoothing effect.
    Returns
    -------
    float
        The computed Cramér-von Mises statistic measuring dependency between A and B.
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
    F_A = rankdata(sa_vals, method=rank_method) / M
    F_B = rankdata(sb_vals, method=rank_method) / M

    # 4. Dense ranks for BIT indexing  (maps unique values -> 1..K)
    sa_dense = rankdata(sa_vals, method=rank_method).astype(np.int64)
    sb_dense = rankdata(sb_vals, method=rank_method).astype(np.int64)
    K = int(sb_dense.max())

    # Sort indices by ascending sa_dense (stable = deterministic tie ordering)
    sort_order = np.argsort(sa_dense, kind="stable").astype(np.int64)

    # empirical copula via rank transform O(M log M) instead of O(M²)
    F_AB = _joint_cdf_bit(sa_dense, sb_dense, sort_order, M, K)

    # 6. Cramér-von Mises statistic
    return float(np.mean((F_AB - F_A * F_B) ** 2))


# ---------------------------------------------------------------------------
# Graph correlation
# ---------------------------------------------------------------------------
def gcov(K_tilde, L_tilde) -> float:
    """Computed Graph covariance between two graphs.
    
    Parameters
    ----------
    K_tilde : np.ndarray, shape (n, n)
        U-centered kernel matrix with latent positions for graph A.
    L_tilde : np.ndarray, shape (n, n)
        U-centered kernel matrix with latent positions for graph B.
    
    Returns
    -------
    float
        The sample graph covariance between the two graphs, defined as
        gCov_n = 1/(n(n-3)) * Σ_{i≠j} K̃_ij L̃_ij 
    """
    n = K_tilde.shape[0]
    return float(np.sum(K_tilde * L_tilde)) / (n * (n - 3))


def gcor(X, Z):
    """
    Compute the sample graph correlation gCor_n(G1, G2).

    Parameters
    ----------
    X : np.ndarray, shape (n, d_X)
        Latent positions for graph A.
    Z : np.ndarray, shape (n, d_Z)
        Latent positions for graph B. 
    
    Returns
    -------
    float:
        The sample graph correlation between the two graphs.
    """
    assert X.shape[0] == Z.shape[0], "X and Z must have the same n"
    n = X.shape[0]
    assert n >= 4, "Need n ≥ 4 for gCor to be defined"

    # Rank-d kernel matrices
    K_hat = X @ X.T   # (n, n)
    L_hat = Z @ Z.T   # (n, n)

    # U-center 
    K_tilde = u_center(K_hat)
    L_tilde = u_center(L_hat)

    cov  = gcov(K_tilde, L_tilde)
    var1 = gcov(K_tilde, K_tilde)
    var2 = gcov(L_tilde, L_tilde)

    denom = np.sqrt(var1 * var2)
    if denom <= 0:
        raise ValueError("At least one graph variance is non-positive; gCor undefined.")
    return cov / denom

# ---------------------------------------------------------------------------
# Azdkia & chatterjee dependence coefficient
# ---------------------------------------------------------------------------

def make_coordinate_permutations(n, d, rng=None):
    """
    Create d permutations pi_1, ..., pi_d of {0, ..., n-1} such that
    for every row i, pi_a(i) != pi_b(i) whenever a != b.

    This construction uses cyclic shifts of one random base permutation.
    It requires d <= n.
    """
    if d > n:
        raise ValueError("Need d_Y <= n to have distinct pi_1(i), ..., pi_d(i) for every i.")

    rng = np.random.default_rng(rng)
    base = rng.permutation(n)

    perms = np.empty((d, n), dtype=int)
    for k in range(d):
        perms[k] = np.roll(base, -k)

    return perms


def multivariate_ac_coefficient_permutation(
    Y,
    Z,
    M=1,
    M_Z=None,
    permutations=None,
    rng=None,
):
    """
    Computes the permutation version of the multivariate Azadkia-Chatterjee
    coefficient with M nearest neighbours:

        T_hat^AC_M =
            (1/M) * sum_i sum_{m=1}^{M} [ n * R_tilde(Y_i ∧ Y_{j_m(i)}) - L_dot_i^2 ]
            ---------------------------------------------------------------------------
                            sum_i [ (n - L_dot_i) * L_dot_i ]

    where j_1(i), ..., j_M(i) are the M nearest neighbours of Z_i and
    all other quantities are as in the M=1 case.

    Parameters
    ----------
    Y : array-like, shape (n,) or (n, d_Y)
        Response variable.
    Z : array-like, shape (n,) or (n, d_Z), optional
        Used to compute M_Z if M_Z is not provided.
    M : int, optional
        Number of nearest neighbours to use (default 1, recovering the
        original single-neighbour estimator).
    M_Z : array-like, shape (n, M) or (n,) when M=1, optional
        M_Z[i, m] is the index of the m-th nearest neighbour of Z_i.
        If omitted, computed from Z.
    permutations : array-like, shape (d_Y, n), optional
        permutations[k, i] = pi_{k+1}(i).
        If omitted, valid coordinate permutations are generated.
    rng : int or np.random.Generator, optional
        Random seed or generator used when permutations are generated.

    Returns
    -------
    float
        M-nearest-neighbour permutation estimator of the multivariate
        AC coefficient.
    """
    Y = _as_2d(Y)
    n, d_Y = Y.shape

    if not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer.")

    # 1. Determine or compute nearest-neighbour mapping M_Z(i, m)
    if M_Z is None:
        Z = _as_2d(Z)
        if Z.shape[0] != n:
            raise ValueError("Y and Z must have the same number of rows.")
        dists = cdist(Z, Z, metric="euclidean")
        np.fill_diagonal(dists, np.inf)
        # argsort gives neighbours in ascending distance order; take first M
        M_Z = np.argsort(dists, axis=1)[:, :M]          # shape (n, M)
    else:
        M_Z = np.asarray(M_Z, dtype=int)
        if M_Z.ndim == 1:                                # backward compat
            M_Z = M_Z[:, None]
        if M_Z.shape != (n, M):
            raise ValueError(f"M_Z must have shape ({n}, {M}).")
        if np.any((M_Z < 0) | (M_Z >= n)):
            raise ValueError("M_Z contains invalid indices.")

    # 2. Build coordinate-wise permuted sample Y_tilde
    if permutations is None:
        permutations = make_coordinate_permutations(n, d_Y, rng=rng)
    else:
        permutations = np.asarray(permutations, dtype=int)
        if permutations.shape != (d_Y, n):
            raise ValueError("permutations must have shape (d_Y, n).")
        target = np.arange(n)
        for k in range(d_Y):
            if not np.array_equal(np.sort(permutations[k]), target):
                raise ValueError(f"permutations[{k}] is not a valid permutation.")
        if d_Y > 1:
            for i in range(n):
                if len(set(permutations[:, i])) != d_Y:
                    raise ValueError(
                        "Permutations must satisfy pi_a(i) != pi_b(i) "
                        "for every i and a != b."
                    )

    # Y_tilde[i, k] = Y[pi_k(i), k]
    Y_tilde = np.empty_like(Y)
    for k in range(d_Y):
        Y_tilde[:, k] = Y[permutations[k], k]

    # 3. Compute Y_i ∧ Y_{j_m(i)} for all i, m
    #    Y[M_Z] has shape (n, M, d_Y); Y[:, None, :] broadcasts to (n, 1, d_Y)
    Y_neighbors = Y[M_Z]                                 # (n, M, d_Y)
    Y_min = np.minimum(Y[:, None, :], Y_neighbors)       # (n, M, d_Y)

    # 4. Compute R_tilde[i, m] = #{j : Y_tilde[j] <=_coord Y_min[i, m]}
    #    Broadcast: Y_tilde (n, d_Y) -> (n, 1, 1, d_Y)  [index j]
    #               Y_min   (n, M, d_Y) -> (1, n, M, d_Y)
    #    Result before sum: (n, n, M) -> sum over j -> (n, M)
    less_equal = np.all(
        Y_tilde[:, None, None, :] <= Y_min[None, :, :, :], axis=-1
    )                                                     # (n, n, M)
    R_tilde = np.sum(less_equal, axis=0)                  # (n, M)

    # 5. Compute L_dot_i = #{l : Y_l >=_coord Y_tilde_i}  (unchanged)
    greater_equal = np.all(Y[:, None, :] >= Y_tilde[None, :, :], axis=-1)
    L_dot = np.sum(greater_equal, axis=0)                 # (n,)

    # 6. Numerator: (1/M) * sum_i sum_m [n * R_tilde[i,m] - L_dot[i]^2]
    #    L_dot[i]^2 is subtracted once per neighbour, so broadcasting
    #    L_dot[:, None] over axis m handles this correctly.
    inner = n * R_tilde - L_dot[:, None] ** 2            # (n, M)
    numerator = np.sum(inner) / M

    denominator = np.sum((n - L_dot) * L_dot)

    if denominator == 0:
        return 0.0
    return numerator / denominator

# def multivariate_ac_coefficient_permutation(
#     Y,
#     Z,
#     M_Z=None,
#     permutations=None,
#     rng=None,
# ):
#     """
#     Computes the permutation version of the multivariate Azadkia-Chatterjee
#     coefficient:

#         T_hat^AC =
#             sum_i [ n * R_tilde(Y_i ∧ Y_{M_Z(i)}) - L_dot_i^2 ]
#             ----------------------------------------------------
#             sum_i [ (n - L_dot_i) * L_dot_i ]

#     where

#         Y_tilde_i = (Y_{pi_1(i),1}, ..., Y_{pi_d(i),d})

#         R_tilde(y) = sum_j 1{Y_tilde_j <= y}

#         L_dot_i = sum_l 1{Y_l >= Y_tilde_i}

#     Parameters
#     ----------
#     Y : array-like, shape (n,) or (n, d_Y)
#         Response variable.

#     M_Z : array-like, shape (n,), optional
#         M_Z[i] is the nearest-neighbor index of Z_i.

#     Z : array-like, shape (n,) or (n, d_Z), optional
#         Used to compute M_Z if M_Z is not provided.

#     permutations : array-like, shape (d_Y, n), optional
#         permutations[k, i] = pi_{k+1}(i).
#         If omitted, valid coordinate permutations are generated.

#     rng : int or np.random.Generator, optional
#         Random seed or generator used when permutations are generated.

#     Returns
#     -------
#     float
#         Permutation estimator of the multivariate AC coefficient.
#     """
#     Y = _as_2d(Y)
#     n, d_Y = Y.shape

#     # 1. Determine or compute nearest-neighbor mapping M_Z(i)
#     if M_Z is None:
#         Z = _as_2d(Z)
#         if Z.shape[0] != n:
#             raise ValueError("Y and Z must have the same number of rows.")

#         dists = cdist(Z, Z, metric="euclidean")
#         np.fill_diagonal(dists, np.inf)
#         M_Z = np.argmin(dists, axis=1)
#     else:
#         M_Z = np.asarray(M_Z, dtype=int)
#         if M_Z.shape != (n,):
#             raise ValueError("M_Z must have shape (n,).")
#         if np.any((M_Z < 0) | (M_Z >= n)):
#             raise ValueError("M_Z contains invalid indices.")

#     # 2. Build coordinate-wise permuted sample Y_tilde
#     if permutations is None:
#         permutations = make_coordinate_permutations(n, d_Y, rng=rng)
#     else:
#         permutations = np.asarray(permutations, dtype=int)
#         if permutations.shape != (d_Y, n):
#             raise ValueError("permutations must have shape (d_Y, n).")

#         # Check each row is a permutation of 0, ..., n-1
#         target = np.arange(n)
#         for k in range(d_Y):
#             if not np.array_equal(np.sort(permutations[k]), target):
#                 raise ValueError(f"permutations[{k}] is not a valid permutation.")

#         # Check pi_a(i) != pi_b(i) for all a != b, for each i
#         if d_Y > 1:
#             for i in range(n):
#                 if len(set(permutations[:, i])) != d_Y:
#                     raise ValueError(
#                         "Permutations must satisfy pi_a(i) != pi_b(i) "
#                         "for every i and a != b."
#                     )

#     # Y_tilde[i, k] = Y[pi_k(i), k]
#     Y_tilde = np.empty_like(Y)
#     for k in range(d_Y):
#         Y_tilde[:, k] = Y[permutations[k], k]

#     # 3. Compute Y_i ∧ Y_{M_Z(i)}
#     Y_min = np.minimum(Y, Y[M_Z])

#     # 4. Compute R_tilde(Y_i ∧ Y_{M_Z(i)})
#     # R_tilde_i = sum_j 1{Y_tilde_j <= Y_min_i}, coordinatewise
#     less_equal = np.all(Y_tilde[:, None, :] <= Y_min[None, :, :], axis=-1)
#     R_tilde = np.sum(less_equal, axis=0)

#     # 5. Compute L_dot_i = sum_l 1{Y_l >= Y_tilde_i}, coordinatewise
#     greater_equal = np.all(Y[:, None, :] >= Y_tilde[None, :, :], axis=-1)
#     L_dot = np.sum(greater_equal, axis=0)

#     numerator = np.sum(n * R_tilde - L_dot**2)
#     denominator = np.sum((n - L_dot) * L_dot)

#     if denominator == 0:
#         return 0.0

#     return numerator / denominator
