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

    return num / den if den != 0 else 0.0


def _rv_squared_gram_norm(values: np.ndarray) -> float:
    """Return the squared Frobenius norm of ``values.T @ values``."""
    gram = values.T @ values
    flattened = gram.ravel()
    return float(flattened.dot(flattened))


def _rv_x_cache(Xhat: np.ndarray) -> tuple[np.ndarray, float]:
    """Precompute the centered X matrix and its RV denominator contribution."""
    centered_x = Xhat - Xhat.mean(axis=0, keepdims=True)
    return centered_x, _rv_squared_gram_norm(centered_x)


def _rv_coefficient_from_x_cache(
    Yhat: np.ndarray,
    centered_x: np.ndarray,
    x_squared_gram_norm: float,
    y_squared_gram_norm: float | None = None,
) -> float:
    """Compute RV while reusing the fixed X-side permutation quantities."""
    centered_y = Yhat - Yhat.mean(axis=0, keepdims=True)
    cross_product = centered_y.T @ centered_x
    flattened_cross_product = cross_product.ravel()
    numerator = flattened_cross_product.dot(flattened_cross_product)

    if y_squared_gram_norm is None:
        y_squared_gram_norm = _rv_squared_gram_norm(centered_y)
    denominator = np.sqrt(y_squared_gram_norm * x_squared_gram_norm)
    return float(numerator / denominator) if denominator != 0 else 0.0


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
def _resolve_cca_gamma(gamma: float | None, n: int) -> float:
    """Validate the X-covariance ridge and resolve its sample-size default."""
    if gamma is None:
        gamma = np.sqrt(n)
    if (
        isinstance(gamma, (bool, np.bool_))
        or not isinstance(gamma, (int, float, np.integer, np.floating))
        or not np.isfinite(gamma)
        or gamma < 0
    ):
        raise ValueError("gamma must be a nonnegative finite scalar or None.")
    return float(gamma)


def _regularized_cca_x_basis(
    Xhat: np.ndarray,
    gamma: float | None = None,
    rcond: float = 1e-10,
) -> np.ndarray:
    """Return the fixed, ridge-whitened left-singular basis for X."""
    n = Xhat.shape[0]
    gamma = _resolve_cca_gamma(gamma, n)
    X = Xhat - Xhat.mean(axis=0, keepdims=True)
    Ux, sx, _ = np.linalg.svd(X, full_matrices=False)
    x_rank = sx > rcond * sx[0] if sx[0] > 0 else np.zeros_like(sx, dtype=bool)
    if not x_rank.any():
        return np.empty((n, 0), dtype=float)

    Ux = Ux[:, x_rank]
    sx = sx[x_rank]
    shrinkage = sx / np.sqrt(sx**2 + n * gamma)
    return Ux * shrinkage


def _first_cca_component_from_x_basis(
    Yhat: np.ndarray,
    regularized_x_basis: np.ndarray,
    rcond: float = 1e-10,
) -> float:
    """Compute the first CCA component using a precomputed X-side basis."""
    Y = Yhat - Yhat.mean(axis=0, keepdims=True)
    Uy, sy, _ = np.linalg.svd(Y, full_matrices=False)
    y_rank = sy > rcond * sy[0] if sy[0] > 0 else np.zeros_like(sy, dtype=bool)
    if not y_rank.any() or regularized_x_basis.shape[1] == 0:
        return 0.0

    whitened_cross_covariance = Uy[:, y_rank].T @ regularized_x_basis
    return float(np.linalg.svd(whitened_cross_covariance, compute_uv=False)[0])


def first_cca_component(
    Xhat: np.ndarray,
    Zhat: np.ndarray,
    rcond: float = 1e-10,
    gamma: float | None = None,
) -> float:
    """
    First canonical correlation with ridge regularization on the second input.

    In the multiple-network test, ``Xhat`` contains the Y-network positions and
    ``Zhat`` contains the concatenated X-network positions. The statistic is
    the largest singular value of

        C_YY^{-1/2} C_YX (C_XX + gamma I)^{-1/2},

    where every covariance uses the ``1/n`` normalization. An SVD formulation
    applies the ridge shrinkage without constructing the potentially large
    X covariance matrix.

    Parameters
    ----------
    Xhat: np.ndarray, shape (n, p)
        Latent positions for the Y network (the first test input).
    Zhat  : np.ndarray, shape (n, q)
        Concatenated latent positions for the X networks. Its sample covariance
        receives the ridge term.

    rcond : threshold for rank truncation (relative to largest singular value)
    gamma : nonnegative float, optional
        Ridge added to the X covariance. Defaults to ``sqrt(n)``.

    Returns
    -------
    float  — first (largest) canonical correlation
    """
    n = Xhat.shape[0]
    assert Zhat.shape[0] == n, "Xhat and Zhat must have the same n"
    regularized_x_basis = _regularized_cca_x_basis(
        Zhat,
        gamma=gamma,
        rcond=rcond,
    )
    return _first_cca_component_from_x_basis(
        Xhat,
        regularized_x_basis,
        rcond=rcond,
    )
