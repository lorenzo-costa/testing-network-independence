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
        
        m_idx, n_idx = _neighbor_maps(z, x=X, n=n, M=M)
        
        if right_neighbor is True:
            return _right_neighbor_coefficient(y[:, 0], z, M)
            
        return _scalar_coefficient(y[:, 0], m_idx, n_idx)
    
    if right_neighbor is True:
        raise ValueError("right_neighbor is only used when Y is univariate.")
    
    if permutation is True:
        perm = _make_permutations(n, d_y, rng=rng)
        y_tilde = np.take_along_axis(y, perm.T, axis=0)
    else:
        y_tilde = y.copy()
        
    m_idx, n_idx = _neighbor_maps(z, x=X, n=n, M=M)

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


def _right_neighbor_coefficient(
    y: np.ndarray,
    x: np.ndarray,
    M: int,
) -> float:
    print('using right')
    """
    Lin-Han M-right-neighbor Chatterjee coefficient.

    xi_{n,M} =
        -2
        + 6 * sum_i sum_m min(R_i, R_{j_m(i)})
          / [(n + 1) * {nM + M(M + 1)/4}].
    """
    n = len(y)
    ranks = rankdata(y, method="max").astype(np.int64)
    right_idx = _right_neighbor_indices(x, M)

    rank_sum = np.minimum(
        ranks[:, None],
        ranks[right_idx],
    ).sum(dtype=np.int64)

    normalizer = (n + 1) * (n * M + M * (M + 1) / 4)

    return float(-2.0 + 6.0 * rank_sum / normalizer)


# NN-indices
def _knn_indices(points: np.ndarray, M: int) -> np.ndarray:
    """Return the M nearest non-self observations for every row."""
    n = len(points)
    tree = cKDTree(points)
    _, candidates = tree.query(points, k=M + 1)
    candidates = np.atleast_2d(candidates)

    neighbors = np.empty((n, M), dtype=np.int64)
    for i, row in enumerate(candidates):
        row = row[row != i]
        if row.size < M:  # Handles duplicate points / distance ties robustly.
            _, row = tree.query(points[i], k=n)
            row = np.asarray(row)
            row = row[row != i]
        neighbors[i] = row[:M]
    return neighbors



def _right_neighbor_indices(x: np.ndarray, M: int) -> np.ndarray:
    """
    Return j_m(i), the m-th right neighbor of x_i.

    For sorted position r_i in {0, ..., n - 1}:
        j_m(i) = order[r_i + m]  if r_i + m < n,
                 i               otherwise.

    Requires distinct scalar x values.
    """
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(
            "Right neighbors require a scalar ordering variable."
        )

    values = x[:, 0]
    n = len(values)
    order = np.argsort(values, kind="stable")

    if np.any(np.diff(values[order]) == 0):
        raise ValueError(
            "Right neighbors require distinct ordering-variable values."
        )

    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)

    targets = position[:, None] + np.arange(1, M + 1)
    indices = np.broadcast_to(
        np.arange(n)[:, None],
        (n, M),
    ).copy()

    valid = targets < n
    indices[valid] = order[targets[valid]]

    return indices

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


def _neighbor_maps(
    z: np.ndarray,
    x,
    *,
    n: int,
    M: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return neighbors in Z/(X,Z), and in X when conditioning is used."""
    if x is None:
        m_idx = _knn_indices(z, M)
        return m_idx, None
    else:
        x = _as_2d(x, name="X", n=n)
        m_idx = _knn_indices(np.hstack((x, z)), M)
        n_idx = _knn_indices(x, M)
        return m_idx, n_idx





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