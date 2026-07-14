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
    K_hat = X @ X.T  # (n, n)
    L_hat = Z @ Z.T  # (n, n)

    # U-center
    K_tilde = u_center(K_hat)
    L_tilde = u_center(L_hat)

    cov = gcov(K_tilde, L_tilde)
    var1 = gcov(K_tilde, K_tilde)
    var2 = gcov(L_tilde, L_tilde)

    denom = np.sqrt(var1 * var2)
    if denom <= 0:
        raise ValueError("At least one graph variance is non-positive; gCor undefined.")
    return cov / denom
