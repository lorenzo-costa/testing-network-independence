"""
Imhof (1961) algorithm
======================
Computes P(Q > x) where Q is a quadratic form in normal variables.

Reference:
    J. P. Imhof, "Computing the Distribution of Quadratic Forms in Normal
    Variables", Biometrika, 48(3/4), 419–426, 1961.

Faithful port of the C++/R implementation in the CompQuadForm R package.
"""

import numpy as np
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Core math helpers (fully vectorised over the lambda/h/delta2 axes)
# ---------------------------------------------------------------------------


def _theta(
    u: float, lam: np.ndarray, h: np.ndarray, x: float, delta2: np.ndarray
) -> float:
    """Phase function θ(u) — eq. (3.2), Imhof 1961 p. 423."""
    lu = lam * u
    lu2 = lu**2
    s = np.sum(h * np.arctan(lu) + delta2 * lu / (1.0 + lu2))
    return 0.5 * s - 0.5 * x * u


def _rho(u: float, lam: np.ndarray, h: np.ndarray, delta2: np.ndarray) -> float:
    """Amplitude function ρ(u) — eq. (3.2), Imhof 1961 p. 423."""
    lu2 = (lam * u) ** 2
    log_rho = np.sum(
        0.25 * h * np.log1p(lu2)  # (1 + (λu)²)^(h/4)
        + 0.5 * delta2 * lu2 / (1.0 + lu2)  # exp(…)
    )
    return np.exp(log_rho)


def _integrand(
    u: float, lam: np.ndarray, h: np.ndarray, x: float, delta2: np.ndarray
) -> float:
    """Integrand sin(θ(u)) / (u · ρ(u)) of eq. (3.2)."""
    return np.sin(_theta(u, lam, h, x, delta2)) / (u * _rho(u, lam, h, delta2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def imhof(
    q,
    lambda_: np.ndarray,
    h: np.ndarray | None = None,
    delta: np.ndarray | None = None,
    epsabs: float = 1e-6,
    epsrel: float = 1e-6,
    limit: int = 10_000,
) -> dict:
    """
    Survival function P(Q > q) of a quadratic form in normal variables.

    Parameters
    ----------
    q : float or array-like
        Value(s) at which to evaluate P(Q > q).
    lambda_ : array-like, shape (m,)
        Eigenvalues of the quadratic form.
    h : array-like, shape (m,), optional
        Degrees-of-freedom weights. Defaults to ones.
    delta : array-like, shape (m,), optional
        Squared non-centrality parameters (δ²). Defaults to zeros.
        Convention matches the CompQuadForm R package.
    epsabs : float
        Absolute error tolerance passed to the integrator.
    epsrel : float
        Relative error tolerance passed to the integrator.
    limit : int
        Maximum number of sub-intervals for adaptive quadrature.

    Returns
    -------
    dict with keys:
        'Qq'     : float or ndarray — P(Q > q)
        'abserr' : float or ndarray — absolute error estimate

    Examples
    --------
    >>> import numpy as np
    >>> # Central chi-squared with 4 df: eigenvalues all 1, h all 1
    >>> res = imhof(6.0, np.ones(4))
    >>> print(round(res['Qq'], 4))   # ≈ 0.1991 (1 - chi2.cdf(6,4))
    """
    lam = np.asarray(lambda_, dtype=float)
    m = len(lam)

    h_ = np.ones(m) if h is None else np.asarray(h, dtype=float)
    d2 = np.zeros(m) if delta is None else np.asarray(delta, dtype=float)

    if len(h_) != m:
        raise ValueError("'lambda_' and 'h' must have the same length.")
    if len(d2) != m:
        raise ValueError("'lambda_' and 'delta' must have the same length.")
    if np.any(d2 < 0):
        raise ValueError("All non-centrality parameters in 'delta' must be >= 0.")

    scalar_input = np.ndim(q) == 0
    q = np.atleast_1d(np.asarray(q, dtype=float))

    Qq = np.empty_like(q)
    abserr = np.empty_like(q)

    for idx, x in np.ndenumerate(q):
        integral, err = quad(
            _integrand,
            0.0,
            np.inf,
            args=(lam, h_, float(x), d2),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        Qq[idx] = 0.5 + integral / np.pi
        abserr[idx] = err

    if scalar_input:
        return {"Qq": float(Qq[0]), "abserr": float(abserr[0])}
    return {"Qq": Qq, "abserr": abserr}


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.stats import chi2

    print("=== Imhof (1961) — smoke tests ===\n")

    # --- Test 1: central chi-squared (all eigenvalues = 1) -----------------
    # Q ~ χ²(k)  ⟺  lambda = [1]*k, h = [1]*k, delta = 0
    for k, x in [(4, 6.0), (10, 15.0), (1, 2.5)]:
        res = imhof(x, np.ones(k))
        expected = 1.0 - chi2.cdf(x, df=k)
        print(
            f"χ²({k}) > {x}: imhof={res['Qq']:.6f}, "
            f"scipy={expected:.6f}, Δ={abs(res['Qq'] - expected):.2e}"
        )

    # --- Test 2: scaled chi-squared (eigenvalues = [2, 3]) -----------------
    # Q = 2*X1 + 3*X2  with X1~χ²(1), X2~χ²(1)
    print()
    lam = np.array([2.0, 3.0])
    for x in [3.0, 7.0, 12.0]:
        res = imhof(x, lam)
        print(
            f"2χ²(1)+3χ²(1) > {x:5.1f}: Qq={res['Qq']:.6f}  (abserr={res['abserr']:.1e})"
        )

    # --- Test 3: vector input -----------------------------------------------
    print()
    xs = np.array([4.0, 6.0, 9.0])
    res = imhof(xs, np.ones(4))
    expected = 1.0 - chi2.cdf(xs, df=4)
    print("Vector input (χ²(4)):")
    for xi, qi, ei in zip(xs, res["Qq"], expected):
        print(f"  x={xi}: imhof={qi:.6f}, scipy={ei:.6f}")
