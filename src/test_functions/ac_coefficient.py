"""Azadkia--Chatterjee-style dependence coefficients.

The public entry point, ``ac_coefficient``, chooses the scalar rank
estimator when Y has one coordinate and the coordinate-permutation
estimator when Y is genuinely multivariate.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata


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


def _validate_neighbors(indices, *, n: int, M: int, name: str) -> np.ndarray:
    """Validate and normalize a supplied neighbor map to shape ``(n, M)``."""
    indices = np.asarray(indices)
    if indices.ndim == 1 and M == 1:
        indices = indices[:, None]
    if indices.shape != (n, M):
        extra = f" or ({n},) when M=1" if M == 1 else ""
        raise ValueError(f"{name} must have shape ({n}, {M}){extra}.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{name} must contain integer indices.")
    if np.any((indices < 0) | (indices >= n)):
        raise ValueError(f"{name} contains invalid indices.")
    if np.any(indices == np.arange(n)[:, None]):
        raise ValueError(f"{name} cannot contain self-neighbors.")
    return indices.astype(np.int64, copy=False)


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


def _resolve_neighbors(points, supplied, *, n: int, M: int, name: str) -> np.ndarray:
    return (
        _knn_indices(points, M)
        if supplied is None
        else _validate_neighbors(supplied, n=n, M=M, name=name)
    )


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


def _resolve_permutations(permutations, *, n: int, d_y: int, rng=None) -> np.ndarray:
    if permutations is None:
        return _make_permutations(n, d_y, rng=rng)

    permutations = np.asarray(permutations)
    if permutations.shape != (d_y, n):
        raise ValueError(f"permutations must have shape ({d_y}, {n}).")
    if not np.issubdtype(permutations.dtype, np.integer):
        raise TypeError("permutations must contain integer indices.")

    target = np.arange(n)
    if not np.all(np.sort(permutations, axis=1) == target):
        raise ValueError("Each row of permutations must be a valid permutation.")
    if np.any(np.diff(np.sort(permutations, axis=0), axis=0) == 0):
        raise ValueError(
            "For each i, permutations[:, i] must contain distinct source indices."
        )
    return permutations.astype(np.int64, copy=False)


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
    M_Z,
    N_X,
    M_XZ,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return neighbors in Z/(X,Z), and in X when conditioning is used."""
    if x is None:
        if N_X is not None or M_XZ is not None:
            raise ValueError("N_X and M_XZ are only valid when X is supplied.")
        return _resolve_neighbors(z, M_Z, n=n, M=M, name="M_Z"), None

    if M_Z is not None:
        raise ValueError("M_Z is only valid when X is None.")

    x = _as_2d(x, name="X", n=n)
    n_idx = _resolve_neighbors(x, N_X, n=n, M=M, name="N_X")
    m_idx = _resolve_neighbors(
        np.hstack((x, z)), M_XZ, n=n, M=M, name="M_XZ"
    )
    return m_idx, n_idx


def _scalar_coefficient(
    y: np.ndarray,
    m_idx: np.ndarray,
    n_idx: np.ndarray | None,
) -> float:
    """The scalar rank estimator, unconditional or conditional."""
    n = len(y)
    m_idx = m_idx[:, 0]
    ranks = rankdata(y, method="max").astype(np.int64)

    if n_idx is None:
        upper_ranks = rankdata(-y, method="max").astype(np.int64)
        numerator = np.sum(n * np.minimum(ranks, ranks[m_idx]) - upper_ranks**2)
        denominator = np.sum(upper_ranks * (n - upper_ranks))
    else:
        n_idx = n_idx[:, 0]
        baseline = np.minimum(ranks, ranks[n_idx])
        numerator = np.sum(np.minimum(ranks, ranks[m_idx]) - baseline)
        denominator = np.sum(ranks - baseline)

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


def ac_coefficient(
    Y,
    Z,
    X=None,
    *,
    M: int = 1,
    M_Z=None,
    N_X=None,
    M_XZ=None,
    permutations=None,
    rng=None,
    block_size: int = 2048,
) -> float:
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
        if M != 1:
            raise ValueError("For scalar Y, use M=1: the rank estimator is 1-NN.")
        if permutations is not None:
            raise ValueError("permutations are only used when Y is multivariate.")
        m_idx, n_idx = _neighbor_maps(
            z, x=X, n=n, M=1, M_Z=M_Z, N_X=N_X, M_XZ=M_XZ
        )
        return _scalar_coefficient(y[:, 0], m_idx, n_idx)

    permutations = _resolve_permutations(
        permutations, n=n, d_y=d_y, rng=rng
    )
    y_tilde = np.take_along_axis(y, permutations.T, axis=0)
    m_idx, n_idx = _neighbor_maps(
        z, x=X, n=n, M=M, M_Z=M_Z, N_X=N_X, M_XZ=M_XZ
    )
    return _multivariate_coefficient(
        y, y_tilde, m_idx, n_idx, block_size=block_size
    )
