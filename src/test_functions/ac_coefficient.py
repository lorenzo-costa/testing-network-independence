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
                   rng=None, block_size=2048):
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
        Response variable(s). If Y is univariate, the scalar rank estimator is
        used; otherwise the multivariate coordinate-permutation estimator is
        used.
    Z : array-like
        Predictor variable(s), used to construct nearest-neighbor maps.
    X : array-like, optional
        Conditioning variable(s). If X is provided, the conditional version of the coefficient is computed. 
    M : int or float, optional
        Number of nearest neighbors. A float smaller than 1 is interpreted as
        an exponent, giving ``round(n**M)`` neighbors. Defaults to 1.
    permutation : bool, optional
        Whether to use the permutation-based estimator for multivariate Y. Default is False.
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

        m_idx, n_idx = _neighbor_maps(z, x=X, n=n, M=M, rng=rng)

        return _scalar_coefficient(y[:, 0], m_idx, n_idx)

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


def asymptotic_unconditional_ac(
    Y, Z, *, M=1, rng=None, block_size=2048
) -> tuple[float, float]:
    """Return the unconditional coefficient and its null variance.

    For scalar Y, the coefficient is the rank statistic; its empirical
    reference distribution is unchanged by a coordinate shuffle. For
    multivariate Y, the coefficient uses the shuffled reference sample.
    The variance is for ``sqrt(n) * coefficient`` and uses the realized
    directed nearest-neighbor graph. A float M below one is resolved to
    ``round(n**M)`` neighbors, as in ``ac_coefficient``. The variance
    calibration is derived for fixed neighbor counts under network--response
    independence; validity for counts growing with n is not established.
    The fallback variance of one follows the studentization convention when
    its empirical components vanish.
    """
    y = _as_2d(Y, name="Y")
    n, d_y = y.shape
    if n < max(d_y, 4):
        raise ValueError("Asymptotic AC calibration requires n >= max(d_Y, 4).")
    z = _as_2d(Z, name="Z", n=n)
    K = _validate_m(M, n)

    if d_y == 1:
        # Shuffling a single coordinate leaves its empirical CDF unchanged.
        # The shuffled-reference statistic is exactly the scalar rank statistic.
        y_tilde = y
        neighbors, _ = _neighbor_maps(z, x=None, n=n, M=K, rng=rng)
        coefficient = _scalar_coefficient(y[:, 0], neighbors, None)
    else:
        permutations = _make_permutations(n, d_y, rng=rng)
        y_tilde = np.take_along_axis(y, permutations.T, axis=0)
        neighbors, _ = _neighbor_maps(z, x=None, n=n, M=K, rng=rng)
        coefficient = _multivariate_coefficient(
            y, y_tilde, neighbors, None, block_size=block_size
        )

    # Empirical reference kernel h_hat(Y_i, Y_j) = F_tilde_n(Y_i wedge Y_j).
    adjacent = _orthant_counts(
        y_tilde, np.minimum(y[:-1], y[1:]),
        relation="le", block_size=block_size,
    ) / n
    skip_one = _orthant_counts(
        y_tilde, np.minimum(y[:-2], y[2:]),
        relation="le", block_size=block_size,
    ) / n
    theta_hat = adjacent.mean()
    a_hat = np.mean(adjacent**2)
    c_hat = np.mean(adjacent[:-1] * skip_one)
    gamma_2_hat = max(c_hat - theta_hat**2, 0.0)
    gamma_1_hat = max(a_hat - 2 * c_hat + theta_hat**2, 0.0)

    rows = np.repeat(np.arange(n), K)
    columns = neighbors.ravel()
    edges = rows * n + columns
    reverse_edges = columns * n + rows
    reciprocal_edges = np.count_nonzero(np.isin(reverse_edges, edges, assume_unique=True))
    indegrees = np.bincount(columns, minlength=n)
    b_n = 1 / K + reciprocal_edges / (n * K**2)
    q_n = np.mean((indegrees / K - 1) ** 2)

    g_n = _orthant_counts(y, y_tilde, relation="ge", block_size=block_size) / n
    s_hat = np.mean(g_n * (1 - g_n))
    variance_numerator = gamma_1_hat * b_n + gamma_2_hat * q_n
    variance = (
        float(variance_numerator / s_hat**2)
        if variance_numerator > 0 and s_hat > 0 else 1.0
    )
    return coefficient, variance
