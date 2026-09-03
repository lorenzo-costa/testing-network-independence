"""Azadkia--Chatterjee-style dependence coefficients.

The public entry point, ``ac_coefficient``, chooses the scalar rank
estimator when Y has one coordinate and the coordinate-permutation
estimator when Y is genuinely multivariate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from ._ac_helpers import (
    _as_2d,
    _neighbor_maps,
    _orthant_counts,
    _validate_m,
    _make_permutations,
)

# Main function
def ac_coefficient(Y, Z, X=None, *, M=1, permutation=False,
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
    M : int or float, optional
        Number of nearest neighbors. A float smaller than 1 is interpreted as
        an exponent, giving ``round(n**M)`` neighbors. Defaults to 1.
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
    y = _as_2d(Y, name="Y")
    n, d_y = y.shape
    if n < 2:
        raise ValueError("At least two observations are required.")

    M = _validate_m(1 if M is None else M, n)

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
