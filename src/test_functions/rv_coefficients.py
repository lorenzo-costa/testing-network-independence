"""RV coefficient helpers."""

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


