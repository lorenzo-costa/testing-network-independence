"""Azadkia--Chatterjee-style dependence coefficients.

The public entry point, ``ac_coefficient``, chooses the scalar rank
estimator when Y has one coordinate and the coordinate-permutation
estimator when Y is genuinely multivariate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from math import isqrt
from ._ac_helpers import (
    _as_2d,
    _neighbor_maps,
    _neighbor_prefix_maps,
    _orthant_counts,
    _validate_aggregate,
    _validate_m,
    _make_permutations,
    _aggregate_coefficients,
)

# Main function
def ac_coefficient(Y, Z, X=None, *, M=None, aggregate=None, permutation=False, 
                   right_neighbor=False, rng=None, block_size=2048):
    """Compute Azadkia-Chatterhee coefficient. 
    
    Depedign on the shape of X and Y different implementations are available:
    - Chatterjee 2021 for univariate X and Y
    - Azadkia& Chatterjee (2021) for multivariate Z and X
    - Deb et al (2020) using more than 1 Nearest Neighbour for Y scalar and X, Z 
    multivariate
    - Some othr guy for Y, Z and X all multivariate using permutations 

    Parameters
    ----------
    Y : array-like
        Response variable(s). If Y is univariate, the scalar rank estimator is used.
    Z : array-like
        Predictor variable(s). If Z is multivariate, the coordinate-permutation estimator is used.
    X : array-like, optional
        Conditioning variable(s). If X is provided, the conditional version of the coefficient is computed. 
    M : int, optional
        Number of nearest neighbors to use. If None, the default is 1 for univariate Y and floor(sqrt(n)) for multivariate Y.
    aggregate : str, optional
        How to aggregate the coefficients over M. Options are "avg" (default), "max", or None (no aggregation). 
    permutation : bool, optional
        Whether to use the permutation-based estimator for multivariate Y. Default is False.
    right_neighbor : bool, optional
        Whether to use the right-neighbor version of the coefficient. Default is False.
    rng : np.random.Generator, optional
        Random number generator. If None, a new generator is created.
    block_size : int, optional
        Block size for orthant count computations. Default is 2048.
    
    Returns
    -------
    float
        The computed Azadkia-Chatterjee coefficient.
    """
    if aggregate is None:
        out = _single_m_ac_coefficient(
            Y,
            Z,
            X=X,
            M=M if M is not None else 1,
            permutation=permutation,
            right_neighbor=right_neighbor,
            rng=rng,
            block_size=block_size,
        )
    elif aggregate in ("avg", "mean", "max"):
        out = _aggregate_m_ac_coefficient(
            Y,
            Z,
            X=X,
            M=M,
            aggregate=aggregate,
            permutation=permutation,
            right_neighbor=right_neighbor,
            rng=rng,
            block_size=block_size,
        )
    else:
        raise ValueError(f"aggregate must be one of 'avg', 'mean', or 'max', got {aggregate}")
        
    return out


def _aggregate_m_ac_coefficient(
    Y,
    Z,
    X=None,
    *,
    aggregate = "avg",
    M=None,
    permutation=False,
    right_neighbor=False,
    rng=None,
    block_size = 2048,
) -> float:
    """Aggregate AC coefficients for every ``M`` in ``1, ..., floor(sqrt(n))``.

    Parameters are the same as :func:`ac_coefficient`, except that ``M`` is
    selected automatically. ``aggregate`` must be either:

    - ``"avg"``: arithmetic mean of the coefficients across M;
    - ``"max"``: largest coefficient across M.

    The implementation computes the first ``floor(sqrt(n))`` neighbors and,
    for multivariate Y, the needed orthant counts only once. It then derives
    every M-specific coefficient from cumulative sums.

    With exact distance ties, it uses one random tie-broken ordering of the
    K-nearest neighbors, where K = floor(sqrt(n)). Each prefix is a valid
    M-nearest-neighbor set. This may differ from repeatedly calling
    :func:`ac_coefficient` with independently redrawn tie breaks for each M.
    """
    aggregate = _validate_aggregate(aggregate)

    y = _as_2d(Y, name="Y")
    n, d_y = y.shape
    if n < 2:
        raise ValueError("At least two observations are required.")
    
    if M is not None:
        max_m = _validate_m(M, n)
    else:
        max_m = isqrt(n)
        
    z = _as_2d(Z, name="Z", n=n)

    if d_y == 1:
        if permutation is True:
            raise ValueError("permutation is only used when Y is multivariate.")

        m_idx, n_idx = _neighbor_prefix_maps(
            z,
            x=X,
            n=n,
            max_m=max_m,
            rng=rng,
            right_neighbor=right_neighbor,
        )

        if right_neighbor is True:
            coefficients = _right_neighbor_coefficients_over_m(y[:, 0], m_idx)
        else:
            coefficients = _scalar_coefficients_over_m(y[:, 0], m_idx, n_idx)

    else:
        if right_neighbor is True:
            raise ValueError("right_neighbor is only used when Y is univariate.")

        if permutation is True:
            perm = _make_permutations(n, d_y, rng=rng)
            y_tilde = np.take_along_axis(y, perm.T, axis=0)
        else:
            y_tilde = y.copy()

        m_idx, n_idx = _neighbor_prefix_maps(
            z,
            x=X,
            n=n,
            max_m=max_m,
            rng=rng,
        )
        coefficients = _multivariate_coefficients_over_m(
            y,
            y_tilde,
            m_idx,
            n_idx,
            block_size=block_size,
        )

    return _aggregate_coefficients(coefficients, aggregate)

def _single_m_ac_coefficient(
    Y,
    Z,
    X=None,
    *,
    M=1,
    permutation=False,
    right_neighbor=False,
    rng=None,
    block_size=2048,
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

        m_idx, n_idx = _neighbor_maps(
            z, x=X, n=n, M=M, rng=rng, right_neighbor=right_neighbor
        )

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


# Coeff calculation functions
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

    normalizer = (n + 1) * (n * M + M * (M + 1) / 4)

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


def _right_neighbor_coefficients_over_m(
    y: np.ndarray,
    m_idx: np.ndarray,
) -> np.ndarray:
    """Return the right-neighbor coefficient for every prefix length."""
    n, max_m = m_idx.shape
    ranks = rankdata(y, method="max").astype(np.int64)
    rank_sums = np.cumsum(
        np.minimum(ranks[:, None], ranks[m_idx]),
        axis=1,
        dtype=np.int64,
    ).sum(axis=0)

    m_values = np.arange(1, max_m + 1, dtype=float)
    normalizers = (n + 1) * (n * m_values + m_values * (m_values + 1) / 4)
    return -2.0 + 6.0 * rank_sums / normalizers


def _multivariate_coefficients_over_m(
    y: np.ndarray,
    y_tilde: np.ndarray,
    m_idx: np.ndarray,
    n_idx: np.ndarray | None,
    *,
    block_size: int,
) -> np.ndarray:
    """Return the multivariate coefficient for every prefix length."""
    n, max_m = m_idx.shape
    divisors = np.arange(1, max_m + 1)

    r_m = _orthant_counts(
        y_tilde,
        np.minimum(y[:, None, :], y[m_idx]),
        relation="le",
        block_size=block_size,
    )
    r_m = np.cumsum(r_m, axis=1, dtype=np.int64) / divisors

    if n_idx is None:
        l_dot = _orthant_counts(
            y,
            y_tilde,
            relation="ge",
            block_size=block_size,
        )
        numerators = np.sum(
            n * r_m - l_dot[:, None] ** 2,
            axis=0,
        )
        denominator = np.sum((n - l_dot) * l_dot)
        if denominator == 0:
            return np.zeros(max_m, dtype=float)
        return numerators / denominator

    r_n = _orthant_counts(
        y_tilde,
        np.minimum(y[:, None, :], y[n_idx]),
        relation="le",
        block_size=block_size,
    )
    r_n = np.cumsum(r_n, axis=1, dtype=np.int64) / divisors
    r_self = _orthant_counts(y_tilde, y, relation="le", block_size=block_size)

    numerators = np.sum(r_m - r_n, axis=0)
    denominators = np.sum(r_self[:, None] - r_n, axis=0)
    return np.divide(
        numerators,
        denominators,
        out=np.zeros(max_m, dtype=float),
        where=denominators != 0,
    )


def _scalar_coefficients_over_m(
    y: np.ndarray,
    m_idx: np.ndarray,
    n_idx: np.ndarray | None,
) -> np.ndarray:
    """Return the scalar coefficient for every prefix length of m_idx."""
    n, max_m = m_idx.shape
    ranks = rankdata(y, method="max").astype(np.int64)
    divisors = np.arange(1, max_m + 1)

    m_score = np.cumsum(
        np.minimum(ranks[:, None], ranks[m_idx]),
        axis=1,
        dtype=np.int64,
    ) / divisors

    if n_idx is None:
        upper_ranks = rankdata(-y, method="max").astype(np.int64)
        numerators = np.sum(
            n * m_score - upper_ranks[:, None] ** 2,
            axis=0,
        )
        denominator = np.sum(upper_ranks * (n - upper_ranks))
        if denominator == 0:
            return np.zeros(max_m, dtype=float)
        return numerators / denominator

    n_score = np.cumsum(
        np.minimum(ranks[:, None], ranks[n_idx]),
        axis=1,
        dtype=np.int64,
    ) / divisors

    numerators = np.sum(m_score - n_score, axis=0)
    denominators = np.sum(ranks[:, None] - n_score, axis=0)
    return np.divide(
        numerators,
        denominators,
        out=np.zeros(max_m, dtype=float),
        where=denominators != 0,
    )


