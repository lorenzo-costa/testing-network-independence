import numpy as np


def u_center(K: np.ndarray) -> np.ndarray:
    """
    Apply U-centering to an n×n kernel matrix.

    For i ≠ j:
        K̃_ij = K_ij - 1/(n-2) * ΣᵥK_iv - 1/(n-2) * ΣᵤK_uj + 1/((n-1)(n-2)) * Σ_{u,v}K_uv
    Diagonal entries are set to 0.
    """
    n = K.shape[0]
    row_sums = K.sum(axis=1, keepdims=True)  # (n, 1)
    col_sums = K.sum(axis=0, keepdims=True)  # (1, n)
    total = K.sum()  # scalar

    K_tilde = K - row_sums / (n - 2) - col_sums / (n - 2) + total / ((n - 1) * (n - 2))
    np.fill_diagonal(K_tilde, 0.0)
    return K_tilde


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


def _gcor_x_cache(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Precompute the centered Gram matrix and graph variance for fixed X."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional latent-position matrix.")
    if X.shape[0] < 4:
        raise ValueError("Need n >= 4 for graph correlation to be defined.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    centered_gram = u_center(X @ X.T)
    variance = gcov(centered_gram, centered_gram)
    return centered_gram, variance


def _gcor_from_x_cache(
    Y: np.ndarray,
    centered_x_gram: np.ndarray,
    x_variance: float,
) -> float:
    """Compute graph correlation while reusing the fixed X-side quantities."""
    Y = np.asarray(Y, dtype=float)
    centered_x_gram = np.asarray(centered_x_gram, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional latent-position matrix.")
    n = Y.shape[0]
    if n < 4:
        raise ValueError("Need n >= 4 for graph correlation to be defined.")
    if centered_x_gram.shape != (n, n):
        raise ValueError("Y and the cached X quantities must have the same n.")
    if not np.isfinite(Y).all():
        raise ValueError("Y must contain only finite values.")

    centered_y_gram = u_center(Y @ Y.T)
    covariance = gcov(centered_y_gram, centered_x_gram)
    y_variance = gcov(centered_y_gram, centered_y_gram)
    denominator = np.sqrt(y_variance * x_variance)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError(
            "At least one graph variance is non-positive; graph correlation "
            "is undefined."
        )
    return covariance / denominator


def gcor(Y, X):
    """
    Compute the sample graph correlation gCor_n(G1, G2).

    Parameters
    ----------
    Y : np.ndarray, shape (n, d_Y)
        Latent positions for the response graph.
    X : np.ndarray, shape (n, d_X)
        Latent positions for the predictor graphs.

    Returns
    -------
    float:
        The sample graph correlation between the two graphs.
    """
    centered_x_gram, x_variance = _gcor_x_cache(X)
    return _gcor_from_x_cache(Y, centered_x_gram, x_variance)
