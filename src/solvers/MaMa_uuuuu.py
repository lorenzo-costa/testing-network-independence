import numpy as np
from scipy.special import expit, logit

# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------
try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from functools import partial as _partial
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_theta(Z, alpha=None, beta=None, X=None):
    """Θ = Z Z^T + α 1^T + 1 α^T + β X"""
    n = Z.shape[0]
    if alpha is None: alpha = np.zeros(n)
    if beta  is None: beta  = 0.0
    if X     is None: X     = np.zeros((n, n))
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


def project_alpha(alpha): return alpha
def project_beta(beta):   return beta


# ---------------------------------------------------------------------------
# [opt-4]  Fused-BLAS NumPy inner loop
# ---------------------------------------------------------------------------

def _pgd_loop_numpy(A, Z, alpha, beta_val, eta_Z, eta_alpha, eta_beta,
                    num_iters, X, has_X):
    """
    NumPy inner loop with fused BLAS matmul (opt-4).

    Key identity
    ------------
    Z @ Z.T + outer(alpha, alpha)
        = [Z | alpha | ones] @ [Z | ones | alpha].T

    One dgemm on (n × k+2) matrices replaces one dgemm on (n × k) matrices
    plus two O(n²) in-place broadcasts.  For n=300, k=2 this saves ~170 µs
    per iteration (~1.47× total).

    When has_X is True, beta*X is added as a single extra pass (unavoidable).
    """
    n, k = Z.shape

    # Pre-allocate extended matrices for fused matmul.
    # Left:  [Z | alpha | ones]
    # Right: [Z | ones  | alpha]
    Zl = np.empty((n, k + 2), order='C')
    Zr = np.empty((n, k + 2), order='C')
    Zl[:, k + 1] = 1.0   # ones column – fixed
    Zr[:, k]     = 1.0   # ones column – fixed

    Theta = np.empty((n, n))
    ones  = np.ones(n)

    for _ in range(num_iters):
        # --- fused Theta build ---
        Zl[:, :k] = Z;  Zl[:, k]     = alpha   # [Z | α | 1]
        Zr[:, :k] = Z;  Zr[:, k + 1] = alpha   # [Z | 1 | α]
        np.dot(Zl, Zr.T, out=Theta)             # Z@Z.T + outer(α,α) in one call

        if has_X:
            Theta += beta_val * X

        # --- sigmoid in-place (4 fused passes, stays in L2 cache) ---
        np.negative(Theta, out=Theta)
        np.exp(Theta, out=Theta)
        Theta += 1.0
        np.reciprocal(Theta, out=Theta)

        # --- residual ---
        np.subtract(A, Theta, out=Theta)

        # --- gradient ascent ---
        Z     += 2.0 * eta_Z    * (Theta @ Z)
        alpha += 2.0 * eta_alpha * (Theta @ ones)
        if has_X:
            beta_val += eta_beta * float(np.sum(Theta * X))

        # --- project Z ---
        Z -= Z.mean(axis=0)

    return Z, alpha, beta_val


# ---------------------------------------------------------------------------
# Optional Numba JIT loop
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @nb.njit(cache=True, fastmath=True)
    def _pgd_loop_numba(A, Z, alpha, beta_val, eta_Z, eta_alpha, eta_beta,
                        num_iters, X, has_X):
        n, k = Z.shape
        Theta = np.empty((n, n))
        ones  = np.ones(n)

        for _ in range(num_iters):
            Theta[:] = Z @ Z.T
            if has_X:
                for i in range(n):
                    ai = alpha[i]
                    for j in range(n):
                        Theta[i, j] += ai + alpha[j] + beta_val * X[i, j]
            else:
                for i in range(n):
                    ai = alpha[i]
                    for j in range(n):
                        Theta[i, j] += ai + alpha[j]

            for i in range(n):
                for j in range(n):
                    Theta[i, j] = 1.0 / (1.0 + np.exp(-Theta[i, j]))

            for i in range(n):
                for j in range(n):
                    Theta[i, j] = A[i, j] - Theta[i, j]

            Z     += 2.0 * eta_Z    * (Theta @ Z)
            alpha += 2.0 * eta_alpha * (Theta @ ones)
            if has_X:
                beta_val += eta_beta * np.sum(Theta * X)

            col_mean = np.zeros(k)
            for j in range(k):
                s = 0.0
                for i in range(n):
                    s += Z[i, j]
                col_mean[j] = s / n
            for j in range(k):
                m = col_mean[j]
                for i in range(n):
                    Z[i, j] -= m

        return Z, alpha, beta_val

    _pgd_loop = _pgd_loop_numba
else:
    _pgd_loop = _pgd_loop_numpy


# ---------------------------------------------------------------------------
# [opt-5]  JAX / XLA backend
# ---------------------------------------------------------------------------

if _HAS_JAX:
    def _make_pgd_loop_jax(has_X: bool):
        """
        Returns a JIT-compiled JAX function for the inner PGD loop.

        Uses jax.lax.while_loop instead of lax.scan, which allows early
        stopping once parameter changes fall below `tol`.  The loop runs
        for at most `max_iters` steps regardless of convergence.

        Convergence criterion
        ---------------------
        delta = ||Z_new - Z||_F / (||Z||_F + 1e-12)
        Iteration stops when  delta <= tol  OR  i >= max_iters.

        Usage
        -----
        _loop = _make_pgd_loop_jax(has_X=False)
        Z_jax, alpha_jax, beta_jax = _loop(
            jnp.array(A), jnp.array(Z), jnp.array(alpha),
            jnp.float32(beta), eta_Z, eta_alpha, eta_beta,
            max_iters=500, X=jnp.array(X), tol=1e-5)
        Z     = np.array(Z_jax)
        alpha = np.array(alpha_jax)
        beta  = float(beta_jax)
        """
        # max_iters no longer needs to be static — while_loop accepts
        # a dynamic bound.  We still JIT the whole function.
        @jit
        def _loop(A, Z, alpha, beta_val, eta_Z, eta_alpha, eta_beta,
                  max_iters, X, tol):

            # ── convergence predicate ────────────────────────────────────
            def cond_fn(carry):
                _Z, _alpha, _beta, i, delta = carry
                return (i < max_iters) & (delta > tol)

            # ── one gradient-ascent step ─────────────────────────────────
            def body_fn(carry):
                Z, alpha, beta_val, i, _delta = carry

                Theta = Z @ Z.T + alpha[:, None] + alpha[None, :]
                if has_X:                          # resolved at trace time
                    Theta = Theta + beta_val * X

                sigma    = jax.nn.sigmoid(Theta)
                residual = A - sigma

                Z_new     = Z     + 2.0 * eta_Z     * (residual @ Z)
                alpha_new = alpha + 2.0 * eta_alpha * residual.sum(axis=1)
                beta_new  = (beta_val + eta_beta * jnp.sum(residual * X)
                             if has_X else beta_val)

                Z_new = Z_new - Z_new.mean(axis=0)

                # relative Frobenius change in Z as stopping signal
                delta = (jnp.linalg.norm(Z_new - Z) /
                         (jnp.linalg.norm(Z) + 1e-12))

                return (Z_new, alpha_new, beta_new, i + 1, delta)

            # ── initialise carry ─────────────────────────────────────────
            # delta starts at +inf so the first iteration always runs
            init_carry = (
                Z, alpha, beta_val,
                jnp.zeros((), dtype=jnp.int32),
                jnp.array(jnp.inf, dtype=jnp.float32),
            )

            Z_out, alpha_out, beta_out, iters_run, _ = jax.lax.while_loop(
                cond_fn, body_fn, init_carry
            )
            return Z_out, alpha_out, beta_out

        return _loop

    # Cache the two variants (with/without X) at import time.
    _pgd_loop_jax_noX = _make_pgd_loop_jax(has_X=False)
    _pgd_loop_jax_X   = _make_pgd_loop_jax(has_X=True)

    def _pgd_loop_jax(A, Z, alpha, beta_val, eta_Z, eta_alpha, eta_beta,
                      num_iters, X, has_X, tol=1e-6):
        """Thin wrapper: numpy → JAX → numpy.

        Parameters
        ----------
        num_iters : int
            Hard upper bound on iterations (replaces the old fixed count).
        tol : float
            Early-stop threshold on the relative change in Z (default 1e-6).
            Pass tol=0.0 to disable early stopping and always run num_iters.
        """
        A_j = jnp.array(A)
        Z_j = jnp.array(Z)
        a_j = jnp.array(alpha)
        b_j = jnp.array(beta_val, dtype=jnp.float32)
        X_j = jnp.array(X)
        tol_j = jnp.array(tol, dtype=jnp.float32)

        fn = _pgd_loop_jax_X if has_X else _pgd_loop_jax_noX
        Z_j, a_j, b_j = fn(
            A_j, Z_j, a_j, b_j,
            eta_Z, eta_alpha, eta_beta,
            num_iters, X_j, tol_j,
        )
        # block_until_ready ensures timing is accurate when benchmarking
        return (np.array(Z_j.block_until_ready()),
                np.array(a_j),
                float(b_j))


# ---------------------------------------------------------------------------
# Main pgd_fit
# ---------------------------------------------------------------------------

def pgd_fit(A, k,
            X=None,
            eta_Z=1e-3,
            eta_alpha=1e-3,
            eta_beta=1e-3,
            num_iters=100,
            Z0=None,
            alpha0=None,
            beta0=None,
            rng=None,
            init='svt',
            tau_init=1e-2,
            M_init=1e-2,
            return_history=False,
            backend='auto'):
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

    Returns
    -------
    Z_hat, alpha_hat, beta_hat  (or with history appended)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    if X is None:
        X = np.zeros((n, n))
    has_X = bool(np.any(X != 0))

    # --- initialise ---
    if init == 'svt':
        alpha, Z, beta = svt_init(A, k, tau=tau_init, M1=M_init, X=X)
    else:
        Z     = rng.standard_normal((n, k)) if Z0 is None else Z0.copy()
        alpha = np.zeros(n)                 if alpha0 is None else alpha0.copy()
        beta  = 0.0                         if beta0 is None else float(beta0)

    if return_history:
        history = [(Z.copy(), alpha.copy(), float(beta))]

    Z     = Z.copy()
    alpha = alpha.copy()
    beta  = float(beta)

    # --- choose backend ---
    if backend == 'auto':
        if _HAS_JAX:
            backend = 'jax'
        elif _HAS_NUMBA:
            backend = 'numba'
        else:
            backend = 'numpy'

    if backend == 'jax' and not _HAS_JAX:
        raise ImportError("JAX not installed. `pip install jax`")
    if backend == 'numba' and not _HAS_NUMBA:
        raise ImportError("Numba not installed. `pip install numba`")

    loop_fn = {
        'numpy': _pgd_loop_numpy,
        'numba': _pgd_loop_numba if _HAS_NUMBA else _pgd_loop_numpy,
        'jax':   _pgd_loop_jax   if _HAS_JAX   else _pgd_loop_numpy,
    }[backend]

    # --- run ---
    if return_history:
        for _ in range(num_iters):
            Z, alpha, beta = loop_fn(
                A, Z, alpha, beta,
                eta_Z, eta_alpha, eta_beta, 1, X, has_X)
            history.append((Z.copy(), alpha.copy(), beta))
        return Z, alpha, beta, history

    Z, alpha, beta = loop_fn(
        A, Z, alpha, beta,
        eta_Z, eta_alpha, eta_beta, num_iters, X, has_X)
    return Z, alpha, beta


def pgd_fit_wrapper(A, k,
                    X=None,
                    eta_Z=1e-3,
                    eta_alpha=1e-3,
                    eta_beta=1e-3,
                    num_iters=500,
                    Z0=None,
                    alpha0=None,
                    beta0=None,
                    rng=None,
                    init='svt',
                    tau_init=1e-2,
                    M_init=4,
                    return_history=False,
                    backend='auto'):
    """Wrapper for pgd_fit returning Z + alpha[:, None]."""
    if return_history:
        Z, alpha, beta, history = pgd_fit(
            A, k, X=X, eta_Z=eta_Z, eta_alpha=eta_alpha, eta_beta=eta_beta,
            num_iters=num_iters, Z0=Z0, alpha0=alpha0, beta0=beta0,
            rng=rng, init=init, tau_init=tau_init, M_init=M_init,
            return_history=True, backend=backend)
        return Z + alpha[:, None], beta, history

    Z, alpha, beta = pgd_fit(
        A, k, X=X, eta_Z=eta_Z, eta_alpha=eta_alpha, eta_beta=eta_beta,
        num_iters=num_iters, Z0=Z0, alpha0=alpha0, beta0=beta0,
        rng=rng, init=init, tau_init=tau_init, M_init=M_init,
        return_history=False, backend=backend)
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
    R    = Theta_hat.sum(axis=1)
    has_cov = bool(np.any(X != 0))

    if not has_cov:
        alpha = R / n
        alpha -= alpha.mean()
        return alpha, 0.0

    RX  = X.sum(axis=1)
    num = np.sum(Theta_hat * X) - (2.0 / n) * np.dot(R, RX)
    den = np.sum(X * X)         - (2.0 / n) * np.dot(RX, RX)
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
    J    = np.eye(n) - np.outer(ones, ones) / n

    U, s, Vt = np.linalg.svd(A)
    mask      = s >= tau
    P_tilde   = (U[:, mask] * s[mask]) @ Vt[mask, :]

    lo = 0.5 * np.exp(-M1)
    hi = 1.0 - lo
    P_hat = np.clip(P_tilde, lo, hi)
    P_sym = (P_hat + P_hat.T) / 2.0
    P_sym = np.clip(P_sym, 1e-6, 1.0 - 1e-6)
    Theta_hat = logit(P_sym)

    if fit_intercept:
        alpha0, beta0 = _fit_additive_model(Theta_hat, X, n)
    else:
        alpha0 = np.zeros(n); beta0 = 0.0

    Theta_resid = Theta_hat - np.add.outer(alpha0, alpha0) - beta0 * X
    R_mat       = J @ Theta_resid @ J

    eigvals, eigvecs = np.linalg.eigh(R_mat)
    eigvals_plus     = np.maximum(eigvals, 0.0)

    idx  = np.argsort(eigvals_plus)[::-1][:k]
    Uk   = eigvecs[:, idx]
    Dk   = np.diag(eigvals_plus[idx])
    Z0   = Uk @ np.sqrt(Dk)

    return alpha0, Z0, beta0
