import numpy as np
from scipy.linalg import blas, norm

# ---------------------------------------------------------------------------
# error functions
# ---------------------------------------------------------------------------
def mse(X, Xhat):
    return ((X - Xhat) ** 2).mean()


def relative_frobenius_norm(X, Xhat, inplace=True):
    if not inplace:
        den = norm(X, "fro")
        return 0 if den == 0 else norm(Xhat - X, "fro") / den

    X_flat = X.ravel()
    Xhat_flat = Xhat.ravel()
    den = blas.dnrm2(X_flat)
    if den == 0:
        return 0
    diff = np.copy(Xhat_flat)
    blas.daxpy(X_flat, diff, a=-1.0)
    return blas.dnrm2(diff) / den

def relative_nuclear_error(X, Xhat):
    """
    Rotation-invariant and more robust than Frobenius.
    Uses the sum of singular values (Nuclear Norm).
    """
    error_matrix = X - Xhat

    # Compute singular values
    s_error = np.linalg.svd(error_matrix, compute_uv=False)
    s_true = np.linalg.svd(X, compute_uv=False)

    return np.sum(s_error) / np.sum(s_true)
