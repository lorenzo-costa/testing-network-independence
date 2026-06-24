"""Azadkia--Chatterjee-style dependence coefficients.

The public entry point, ``ac_coefficient``, chooses the scalar rank
estimator when Y has one coordinate and the coordinate-permutation
estimator when Y is genuinely multivariate.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

# Main function
def ac_coefficient(
    Y,
    Z,
    X=None,
    *,
    M = 1,
    permutation=False,
    right_neighbor=False,
    rng=None,
    block_size = 2048,
):
    """Compute an unconditional or conditional Azadkia--Chatterjee coefficient.

    ``Y`` with one coordinate uses the scalar rank estimator. A genuinely
    multivariate ``Y`` uses the coordinate-permutation estimator. ``X=None``
    selects the unconditional version; supplying ``X`` selects the conditional
    version.
    """
    
    y = _as_2d(Y, name="Y")
    n, d_y = y.shape
    if n < 2:
        raise ValueError("At least two observations are required.")
    
    M = _validate_m(M, n)
    
    z = _as_2d(Z, name="Z", n=n)

    if d_y == 1:
        if permutation is True:
            raise ValueError("permutation is only used when Y is multivariate.")
        
        m_idx, n_idx = _neighbor_maps(z, x=X, n=n, M=M, rng=rng, right_neighbor=right_neighbor)
        
        if right_neighbor is True:
            return _right_neighbor_coefficient(y[:, 0], m_idx, rng=rng)
            
        return _scalar_coefficient(y[:, 0], m_idx, n_idx)
    
    if right_neighbor is True:
        raise ValueError("right_neighbor is only used when Y is univariate.")
    
    if permutation is True:
        perm = _make_permutations(n, d_y, rng=rng)
        y_tilde = np.take_along_axis(y, perm.T, axis=0)
    else:
        y_tilde = y.copy()
        
    m_idx, n_idx = _neighbor_maps(z, x=X, n=n, M=M, rng=rng)

    return _multivariate_coefficient(y, y_tilde, m_idx, n_idx, block_size=block_size)


#Coeff calculation functions
def _scalar_coefficient(
    y: np.ndarray,
    m_idx: np.ndarray,
    n_idx: np.ndarray | None,
) -> float:
    """Scalar rank estimator, averaged over M nearest neighbors."""
    n = len(y)
    ranks = rankdata(y, method="max").astype(np.int64)

    m_score = np.minimum(ranks[:, None], ranks[m_idx]).mean(axis=1)

    if n_idx is None:
        upper_ranks = rankdata(-y, method="max").astype(np.int64)

        numerator = np.sum(n * m_score - upper_ranks**2)
        denominator = np.sum(upper_ranks * (n - upper_ranks))

    else:
        n_score = np.minimum(ranks[:, None], ranks[n_idx]).mean(axis=1)

        numerator = np.sum(m_score - n_score)
        denominator = np.sum(ranks - n_score)

    return 0.0 if denominator == 0 else float(numerator / denominator)

def _right_neighbor_coefficient(
    y: np.ndarray,
    M_idx: np.ndarray,
    *,
    rng=None,
) -> float:
    n = len(y)
    M = M_idx.shape[1]
    ranks = rankdata(y, method="max").astype(np.int64)

    rank_sum = np.minimum(
        ranks[:, None],
        ranks[M_idx],
    ).sum(dtype=np.int64)

    normalizer = (n + 1) * (
        n * M + M * (M + 1) / 4
    )

    return float(-2.0 + 6.0 * rank_sum / normalizer)
    

def _multivariate_coefficient(
    y: np.ndarray,
    y_tilde: np.ndarray,
    m_idx: np.ndarray,
    n_idx: np.ndarray | None,
    *,
    block_size: int,
) -> float:
    """The coordinate-permutation estimator, unconditional or conditional."""
    n = len(y)
    r_m = _orthant_counts(
        y_tilde,
        np.minimum(y[:, None, :], y[m_idx]),
        relation="le",
        block_size=block_size,
    )

    if n_idx is None:
        l_dot = _orthant_counts(
            y,
            y_tilde,
            relation="ge",
            block_size=block_size,
        )
        numerator = np.sum(n * r_m - l_dot[:, None] ** 2) / m_idx.shape[1]
        denominator = np.sum((n - l_dot) * l_dot)
    else:
        r_n = _orthant_counts(
            y_tilde,
            np.minimum(y[:, None, :], y[n_idx]),
            relation="le",
            block_size=block_size,
        )
        r_self = _orthant_counts(y_tilde, y, relation="le", block_size=block_size)
        r_m = r_m.mean(axis=1)
        r_n = r_n.mean(axis=1)
        numerator = np.sum(r_m - r_n)
        denominator = np.sum(r_self - r_n)

    return 0.0 if denominator == 0 else float(numerator / denominator)



# NN-indices
def _neighbor_maps(
    z,
    x,
    *,
    n: int,
    M: int,
    rng=None,
    right_neighbor=False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return neighbors in Z/(X,Z), and in X when conditioning is used."""
    if right_neighbor is True:
        if x is not None:
            raise ValueError("right_neighbor is only used when X is None.")
        m_idx = _right_neighbor_indices(z, M, rng=rng)
        return m_idx, None
    if x is None:
        m_idx = _knn_indices(z, M, rng=rng)
        return m_idx, None
    else:
        x = _as_2d(x, name="X", n=n)
        m_idx = _knn_indices(np.hstack((x, z)), M, rng=rng)
        n_idx = _knn_indices(x, M, rng=rng)
        return m_idx, n_idx
    
def _knn_indices(
    points: np.ndarray,
    M: int,
    *,
    rng=None,
) -> np.ndarray:
    """
    Return the M nearest non-self neighbors for every row.

    Distance ties at the M-th cutoff are broken uniformly at random.
    Passing the same seed gives reproducible results, independent of the
    order returned by cKDTree for tied points.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)

    if not 1 <= M < n:
        raise ValueError("Require 1 <= M < n.")

    generator = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )

    tree = cKDTree(points)

    # Query M+1 because one returned point is normally self.
    distances, indices = tree.query(points, k=M + 1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    neighbors = np.empty((n, M), dtype=np.int64)

    for i in range(n):
        # Remove self, regardless of where cKDTree placed it.
        mask = indices[i] != i
        d = distances[i][mask]
        idx = indices[i][mask]

        # In duplicate-point cases, self might not be included among M+1
        # returned entries. Re-query more broadly in that rare case.
        if len(idx) < M:
            d, idx = tree.query(points[i], k=n)
            d = np.asarray(d)
            idx = np.asarray(idx)

            mask = idx != i
            d = d[mask]
            idx = idx[mask]

        cutoff = d[M - 1]

        # Radius is nudged upward so points exactly at cutoff are included.
        radius = np.nextafter(cutoff, np.inf)
        candidates = np.asarray(tree.query_ball_point(points[i], radius))

        candidates = candidates[candidates != i]

        candidate_distances = np.linalg.norm(
            points[candidates] - points[i],
            axis=1,
        )

        # Tolerance is needed because floating-point distance computations
        # can differ slightly between KD-tree traversal and direct norm.
        atol = np.finfo(float).eps * max(1.0, cutoff) * 16

        strictly_closer = candidates[candidate_distances < cutoff - atol]
        tied = candidates[np.abs(candidate_distances - cutoff) <= atol]

        needed = M - len(strictly_closer)

        if needed < 0:
            # Numerical guard: retain exactly the M closest, with seeded
            # random ordering only within nearly equal distances.
            jitter = generator.random(len(candidates))
            order = np.lexsort((jitter, candidate_distances))
            neighbors[i] = candidates[order[:M]]
            continue

        if len(tied) < needed:
            # Extremely unusual numerical mismatch; recover with an exact
            # local sort, still avoiding a full n-by-n matrix.
            jitter = generator.random(len(candidates))
            order = np.lexsort((jitter, candidate_distances))
            neighbors[i] = candidates[order[:M]]
            continue

        if needed == 0:
            neighbors[i] = strictly_closer[:M]
        else:
            chosen_ties = generator.choice(tied, size=needed, replace=False)
            neighbors[i] = np.concatenate((strictly_closer, chosen_ties))

    return neighbors

def _right_neighbor_indices(
    z: np.ndarray,
    M: int,
    rng=None,
) -> np.ndarray:
    """
    Return the m-th right neighbor of each scalar z_i.

    Ties in z are broken uniformly at random, reproducibly when rng is given.
    """

    if z.ndim != 2 or z.shape[1] != 1:
        raise ValueError(
            "Right neighbors require a scalar ordering variable."
        )

    values = z[:, 0]
    n = len(values)

    generator = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )

    # Primary sort key: values.
    # Secondary sort key: random numbers, used only within ties.
    tie_breaker = generator.random(n)
    order = np.lexsort((tie_breaker, values))

    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)

    targets = position[:, None] + np.arange(1, M + 1)

    # Boundary convention: if the m-th right neighbor does not exist,
    # map the observation to itself.
    neighbors = np.broadcast_to(
        np.arange(n)[:, None],
        (n, M),
    ).copy()

    valid = targets < n
    neighbors[valid] = order[targets[valid]]

    return neighbors

# Helpers
def _as_2d(values, *, name: str, n: int | None = None) -> np.ndarray:
    """Return a finite array with shape ``(n, d)``."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim != 2:
        raise ValueError(f"{name} must have shape (n,) or (n, d).")

    if values.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column.")
    if n is not None and values.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values

def _validate_m(M: int, n: int) -> int:
    if not isinstance(M, (int, np.integer)) or isinstance(M, bool) or M < 1:
        raise ValueError("M must be a positive integer.")
    if M >= n:
        raise ValueError("M must be smaller than n.")
    return int(M)

def _make_permutations(n: int, d_y: int, rng=None) -> np.ndarray:
    """Make coordinate permutations with distinct source rows per observation."""
    if d_y > n:
        raise ValueError(
            "The multivariate permutation construction requires d_Y <= n."
        )

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    base = generator.permutation(n)
    shifts = generator.choice(n, size=d_y, replace=False)
    return np.array([np.roll(base, -shift) for shift in shifts], dtype=np.int64)

def _orthant_counts(
    sample: np.ndarray,
    thresholds: np.ndarray,
    *,
    relation: str,
    block_size: int,
) -> np.ndarray:
    """Count sample rows lying below or above each coordinatewise threshold."""
    if block_size < 1:
        raise ValueError("block_size must be positive.")

    thresholds = np.asarray(thresholds)
    if sample.ndim != 2 or thresholds.shape[-1] != sample.shape[1]:
        raise ValueError("Incompatible sample and threshold shapes.")

    compare = np.less_equal if relation == "le" else np.greater_equal
    flat = thresholds.reshape(-1, sample.shape[1])
    counts = np.empty(len(flat), dtype=np.int64)

    for start in range(0, len(flat), block_size):
        stop = min(start + block_size, len(flat))
        counts[start:stop] = np.sum(
            np.all(compare(sample[:, None, :], flat[None, start:stop, :]), axis=-1),
            axis=0,
        )
    return counts.reshape(thresholds.shape[:-1])