import numpy as np
from scipy.stats import rankdata
from numba import njit


# ---------------------------------------------------------------------------
# CvM computations
# ---------------------------------------------------------------------------
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
