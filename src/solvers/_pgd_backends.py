"""NumPy, Numba and JAX PGD kernels, with their original stopping rules."""

import numpy as np

try:
    import numba as nb

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _pgd_loop_numpy(
    A, Z, alpha, beta_val, eta_Z, eta_alpha, eta_beta, num_iters, X, has_X, tol
):
    """
    NumPy inner loop with fused BLAS matmul (opt-4).

    Key identity
    ------------
    Z @ Z.T + alpha[:, None] + alpha[None, :]
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
    Zl = np.empty((n, k + 2), order="C")
    Zr = np.empty((n, k + 2), order="C")
    Zl[:, k + 1] = 1.0  # ones column – fixed
    Zr[:, k] = 1.0  # ones column – fixed

    Theta = np.empty((n, n))
    ones = np.ones(n)

    for _ in range(num_iters):
        # --- fused Theta build ---
        Zl[:, :k] = Z
        Zl[:, k] = alpha  # [Z | α | 1]
        Zr[:, :k] = Z
        Zr[:, k + 1] = alpha  # [Z | 1 | α]
        np.dot(Zl, Zr.T, out=Theta)  # Z@Z.T + α_i + α_j in one call

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
        Z += 2.0 * eta_Z * (Theta @ Z)
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
    def _pgd_loop_numba(
        A,
        Z,
        alpha,
        beta_val,
        eta_Z,
        eta_alpha,
        eta_beta,
        num_iters,
        X,
        has_X,
        tol,
    ):
        n, k = Z.shape
        Theta = np.empty((n, n))
        ones = np.ones(n)
        Z_previous = np.empty_like(Z)

        for _ in range(num_iters):
            Z_previous[:] = Z
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

            Z += 2.0 * eta_Z * (Theta @ Z)
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

            if tol > 0.0:
                change_squared = 0.0
                previous_squared = 0.0
                for i in range(n):
                    for j in range(k):
                        difference = Z[i, j] - Z_previous[i, j]
                        change_squared += difference * difference
                        previous_squared += Z_previous[i, j] * Z_previous[i, j]
                delta = np.sqrt(change_squared) / (np.sqrt(previous_squared) + 1e-12)
                if delta <= tol:
                    break

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
        def _loop(A, Z, alpha, beta_val, eta_Z, eta_alpha, eta_beta, max_iters, X, tol):
            # ── convergence predicate ────────────────────────────────────
            def cond_fn(carry):
                _Z, _alpha, _beta, i, delta = carry
                return (i < max_iters) & (delta > tol)

            # ── one gradient-ascent step ─────────────────────────────────
            def body_fn(carry):
                Z, alpha, beta_val, i, _delta = carry

                Theta = Z @ Z.T + alpha[:, None] + alpha[None, :]
                if has_X:  # resolved at trace time
                    Theta = Theta + beta_val * X

                sigma = jax.nn.sigmoid(Theta)
                residual = A - sigma

                Z_new = Z + 2.0 * eta_Z * (residual @ Z)
                alpha_new = alpha + 2.0 * eta_alpha * residual.sum(axis=1)
                beta_new = (
                    beta_val + eta_beta * jnp.sum(residual * X) if has_X else beta_val
                )

                Z_new = Z_new - Z_new.mean(axis=0)

                # relative Frobenius change in Z as stopping signal
                delta = jnp.linalg.norm(Z_new - Z) / (jnp.linalg.norm(Z) + 1e-12)

                return (Z_new, alpha_new, beta_new, i + 1, delta)

            # ── initialise carry ─────────────────────────────────────────
            # delta starts at +inf so the first iteration always runs
            init_carry = (
                Z,
                alpha,
                beta_val,
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
    _pgd_loop_jax_X = _make_pgd_loop_jax(has_X=True)

    def _pgd_loop_jax(
        A, Z, alpha, beta_val, eta_Z, eta_alpha, eta_beta, num_iters, X, has_X, tol
    ):
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
            A_j,
            Z_j,
            a_j,
            b_j,
            eta_Z,
            eta_alpha,
            eta_beta,
            num_iters,
            X_j,
            tol_j,
        )
        # block_until_ready ensures timing is accurate when benchmarking
        return (np.array(Z_j.block_until_ready()), np.array(a_j), float(b_j))
