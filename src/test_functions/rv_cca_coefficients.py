"""Canonical correlation coefficient used by the CCA permutation test."""

import numpy as np


def first_cca_component(
    Xhat: np.ndarray, Zhat: np.ndarray, rcond: float = 1e-10
) -> float:
    """Return the largest unregularized canonical correlation.

    Center both matrices and compute the largest singular value of the
    cross-product of their orthonormal column-space bases. Directions below
    ``rcond`` times the leading singular value are discarded.
    """
    Xhat = np.asarray(Xhat, dtype=float)
    Zhat = np.asarray(Zhat, dtype=float)
    if Xhat.ndim != 2 or Zhat.ndim != 2 or Xhat.shape[0] != Zhat.shape[0]:
        raise ValueError("Xhat and Zhat must be 2D arrays with the same number of rows.")
    if Xhat.shape[1] == 0 or Zhat.shape[1] == 0:
        raise ValueError("Xhat and Zhat must each have at least one column.")

    X = Xhat - Xhat.mean(axis=0, keepdims=True)
    Z = Zhat - Zhat.mean(axis=0, keepdims=True)
    Ux, sx, _ = np.linalg.svd(X, full_matrices=False)
    Uz, sz, _ = np.linalg.svd(Z, full_matrices=False)

    rank_x = sx > rcond * sx[0] if sx[0] > 0 else np.zeros_like(sx, dtype=bool)
    rank_z = sz > rcond * sz[0] if sz[0] > 0 else np.zeros_like(sz, dtype=bool)
    if not rank_x.any() or not rank_z.any():
        return 0.0

    return float(np.linalg.svd(Ux[:, rank_x].T @ Uz[:, rank_z], compute_uv=False)[0])
