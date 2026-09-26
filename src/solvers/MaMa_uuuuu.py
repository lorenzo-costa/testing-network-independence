"""PGD initialization and orchestration (Ma & Ma); kernels live in _pgd_backends."""

import numpy as np
from scipy.special import logit

# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------
from ._pgd_backends import (
    _HAS_JAX,
    _HAS_NUMBA,
    _pgd_loop_numpy,
    _pgd_loop as _pgd_loop,
)

if _HAS_NUMBA:
    from ._pgd_backends import _pgd_loop_numba
if _HAS_JAX:
    from ._pgd_backends import _pgd_loop_jax


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def compute_theta(Z, alpha=None, beta=None, X=None):
    """Θ = Z Z^T + α 1^T + 1 α^T + β X"""
    n = Z.shape[0]
    if alpha is None:
        alpha = np.zeros(n)
    if beta is None:
        beta = 0.0
    if X is None:
        X = np.zeros((n, n))
    return Z @ Z.T + np.add.outer(alpha, alpha) + beta * X


def project_Z(Z, M=None):
    """Centre columns then (optionally) clip rows to ball of radius M^(1/3)."""
    Z = Z - Z.mean(axis=0)
    if M is not None:
        radius = M ** (1.0 / 3.0)
        row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
        scale = np.where(row_norms > radius, radius / row_norms, 1.0)
        Z = Z * scale
    return Z


def project_alpha(alpha):
    return alpha


def project_beta(beta):
    return beta


def pgd_fit(
    A,
    k,
    X=None,
    eta_Z=1e-3,
    eta_alpha=1e-3,
    eta_beta=1e-3,
    num_iters=100,
    Z0=None,
    alpha0=None,
    beta0=None,
    rng=None,
    init="svt",
    tau_init=1e-2,
    M_init=1e-2,
    return_history=False,
    backend="auto",
    tol=1e-6,
):
    """
    Projected Gradient Descent for latent space network model.

    Parameters
    ----------
    A         : (n, n) adjacency matrix
    X         : (n, n) covariate matrix (optional)
    k         : latent space dimension
    eta_Z     : step size for Z
    eta_alpha : step size for alpha
    eta_beta  : step size for beta
    num_iters : number of gradient steps
    Z0        : initial Z  (n × k), random if None
    alpha0    : initial alpha (n,), zeros if None
    beta0     : initial beta (scalar), zero if None
    rng       : np.random.Generator
    init      : 'svt' (default) or 'random'
    tau_init  : SVT threshold for svt initialisation
    M_init    : M parameter for svt initialisation
    return_history : return (Z, alpha, beta, history) if True
    backend   : 'auto' | 'numpy' | 'numba' | 'jax'
                'auto' picks jax > numba > numpy (in that order of preference).
    tol       : nonnegative float
                Relative change threshold for early stopping in the Numba and
                JAX backends. Set to zero to run exactly ``num_iters`` steps.

    Returns
    -------
    Z_hat, alpha_hat, beta_hat  (or with history appended)
    """
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(tol, (bool, np.bool_)) or not np.isscalar(tol):
        raise ValueError("tol must be a nonnegative finite scalar.")
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a nonnegative finite scalar.")

    n = A.shape[0]
    if X is None:
        X = np.zeros((n, n))
    has_X = bool(np.any(X != 0))

    # --- initialise ---
    if init == "svt":
        alpha, Z, beta = svt_init(A, k, tau=tau_init, M1=M_init, X=X)
    else:
        Z = rng.standard_normal((n, k)) if Z0 is None else Z0.copy()
        alpha = np.zeros(n) if alpha0 is None else alpha0.copy()
        beta = 0.0 if beta0 is None else float(beta0)

    if return_history:
        history = [(Z.copy(), alpha.copy(), float(beta))]

    Z = Z.copy()
    alpha = alpha.copy()
    beta = float(beta)

    # --- choose backend ---
    if backend == "auto":
        if _HAS_JAX:
            backend = "jax"
        elif _HAS_NUMBA:
            backend = "numba"
        else:
            backend = "numpy"

    if backend == "jax" and not _HAS_JAX:
        raise ImportError("JAX not installed. `pip install jax`")
    if backend == "numba" and not _HAS_NUMBA:
        raise ImportError("Numba not installed. `pip install numba`")

    loop_fn = {
        "numpy": _pgd_loop_numpy,
        "numba": _pgd_loop_numba if _HAS_NUMBA else _pgd_loop_numpy,
        "jax": _pgd_loop_jax if _HAS_JAX else _pgd_loop_numpy,
    }[backend]

    # --- run ---
    if return_history:
        for _ in range(num_iters):
            previous_Z = Z.copy()
            Z, alpha, beta = loop_fn(
                A,
                Z,
                alpha,
                beta,
                eta_Z,
                eta_alpha,
                eta_beta,
                1,
                X,
                has_X,
                0.0,
            )
            history.append((Z.copy(), alpha.copy(), beta))
            if backend in {"numba", "jax"} and tol > 0:
                delta = np.linalg.norm(Z - previous_Z) / (
                    np.linalg.norm(previous_Z) + 1e-12
                )
                if delta <= tol:
                    break
        return Z, alpha, beta, history

    Z, alpha, beta = loop_fn(
        A,
        Z,
        alpha,
        beta,
        eta_Z,
        eta_alpha,
        eta_beta,
        num_iters,
        X,
        has_X,
        tol if backend in {"numba", "jax"} else 0.0,
    )
    return Z, alpha, beta


def pgd_fit_wrapper(
    A,
    k,
    X=None,
    eta_Z=1e-3,
    eta_alpha=1e-3,
    eta_beta=1e-3,
    num_iters=500,
    Z0=None,
    alpha0=None,
    beta0=None,
    rng=None,
    init="svt",
    tau_init=1e-2,
    M_init=4,
    return_history=False,
    backend="auto",
    tol=1e-6,
):
    """Wrapper for pgd_fit returning Z + alpha[:, None]."""

    fitted = pgd_fit(
        A,
        k,
        X=X,
        eta_Z=eta_Z,
        eta_alpha=eta_alpha,
        eta_beta=eta_beta,
        num_iters=num_iters,
        Z0=Z0,
        alpha0=alpha0,
        beta0=beta0,
        rng=rng,
        init=init,
        tau_init=tau_init,
        M_init=M_init,
        return_history=return_history,
        backend=backend,
        tol=tol,
    )
    if return_history:
        Z, alpha, beta, history = fitted
        return Z + alpha[:, None], beta, history
    Z, alpha, beta = fitted
    return Z + alpha[:, None], beta


# ---------------------------------------------------------------------------
# SVT initialisation  (Algorithm 3 from Ma & Ma 2020)
# ---------------------------------------------------------------------------


def _fit_additive_model(Theta_hat, X, n):
    """
    O(n²) closed-form solution for:
        min_{alpha, beta}  ||Θ - (α_i + α_j + β X_{ij})||_F²
        s.t.  Σ_i α_i = 0
    """
    R = Theta_hat.sum(axis=1)
    has_cov = bool(np.any(X != 0))

    if not has_cov:
        alpha = R / n
        alpha -= alpha.mean()
        return alpha, 0.0

    RX = X.sum(axis=1)
    num = np.sum(Theta_hat * X) - (2.0 / n) * np.dot(R, RX)
    den = np.sum(X * X) - (2.0 / n) * np.dot(RX, RX)
    beta = (num / den) if abs(den) > 1e-12 else 0.0

    alpha = (R - beta * RX) / n
    alpha -= alpha.mean()
    return alpha, beta


def svt_init(A, k, tau, M1, X=None, fit_intercept=True):
    """
    Algorithm 3: SVT-based initialisation.

    Parameters
    ----------
    A   : (n, n) adjacency matrix
    X   : (n, n) covariate matrix (or None)
    k   : latent space dimension
    tau : singular value threshold
    M1  : controls clipping interval [0.5 exp(-M1), 1 - 0.5 exp(-M1)]

    Returns
    -------
    alpha0 : (n,)   initial node effects
    Z0     : (n, k) initial latent positions
    beta0  : scalar initial covariate coefficient
    """
    n = A.shape[0]
    if X is None:
        X = np.zeros((n, n))

    ones = np.ones(n)
    J = np.eye(n) - np.outer(ones, ones) / n

    U, s, Vt = np.linalg.svd(A)
    mask = s >= tau
    P_tilde = (U[:, mask] * s[mask]) @ Vt[mask, :]

    lo = 0.5 * np.exp(-M1)
    hi = 1.0 - lo
    P_hat = np.clip(P_tilde, lo, hi)
    P_sym = (P_hat + P_hat.T) / 2.0
    P_sym = np.clip(P_sym, 1e-6, 1.0 - 1e-6)
    Theta_hat = logit(P_sym)

    if fit_intercept:
        alpha0, beta0 = _fit_additive_model(Theta_hat, X, n)
    else:
        alpha0 = np.zeros(n)
        beta0 = 0.0

    Theta_resid = Theta_hat - np.add.outer(alpha0, alpha0) - beta0 * X
    R_mat = J @ Theta_resid @ J

    eigvals, eigvecs = np.linalg.eigh(R_mat)
    eigvals_plus = np.maximum(eigvals, 0.0)

    idx = np.argsort(eigvals_plus)[::-1][:k]
    Uk = eigvecs[:, idx]
    Dk = np.diag(eigvals_plus[idx])
    Z0 = Uk @ np.sqrt(Dk)

    return alpha0, Z0, beta0
