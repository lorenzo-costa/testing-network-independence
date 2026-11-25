"""Gaussian copula with scalar continuous conditioning covariates."""

import numpy as np
from scipy import stats

from .copula_sampler import CopulaGenerator
from .conditional_copula import ConditionalIndependenceCopulaSampler


class ContinuousConditionalCopulaSampler:
    """Sample (Z, Y, X), or fresh Y conditional on fixed continuous X.

    ``correlation_zx`` and ``correlation_yx`` are scalar (broadcast) or
    vectors of lengths k and ky. ``conditional_correlation_zy`` is a scalar
    (broadcast) or a (k, ky) matrix of residual correlations given X.
    Residual coordinates within Z and within Y are independent. The entire
    residual correlation matrix must be positive definite. Consequently a
    broadcast cross correlation can have stricter bounds than (-1, 1).

    Marginals use the existing string specifications, with exactly the keys
    x, z, y. Each block shares its marginal across columns. All default to
    standard Gaussian. No empirical centering is applied: it would change
    the specified conditional law. Only Gaussian copulas are supported.
    """

    def __init__(self, n, k, ky=1, marginals=None, correlation_zx=0,
                 correlation_yx=0, conditional_correlation_zy=0,
                 copula_model="gaussian", rng=None, center_latent=False,
                 rho=None, **kwargs):
        positive_integer = ConditionalIndependenceCopulaSampler._positive_integer
        self.n = positive_integer(n, "n")
        self.k = positive_integer(k, "k")
        self.ky = positive_integer(ky, "ky")
        if copula_model != "gaussian":
            raise ValueError("Continuous conditional copulas require copula_model='gaussian'.")
        if center_latent:
            raise ValueError("center_latent must be False for continuous conditional copulas.")
        if rho is not None and np.any(np.asarray(rho) != 0):
            raise ValueError("Use conditional_correlation_zy instead of rho (omit rho or set it to zero).")
        for name in ("C", "means", "covariances", "marginals_by_class",
                     "class_probabilities", "p", "column_covariance",
                     "column_covariance_y", "cross_covariance"):
            if name in kwargs:
                raise ValueError(f"{name} is not supported by continuous conditional copulas.")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.marginals = ({key: "gaussian" for key in ("x", "z", "y")}
                          if marginals is None else dict(marginals))
        if set(self.marginals) != {"x", "z", "y"}:
            raise ValueError("marginals must contain exactly the keys 'x', 'z', and 'y'.")
        self._marginals = {}
        for key, spec in self.marginals.items():
            distribution = CopulaGenerator.parse_marginal(spec)
            # Check the scalar continuous CDF/PPF contract, including parameters.
            try:
                quantiles = np.asarray(distribution.ppf([0.25, 0.5, 0.75]))
                valid = (quantiles.shape == (3,) and np.isfinite(quantiles).all()
                         and np.all(np.diff(quantiles) > 0)
                         and np.allclose(distribution.cdf(quantiles), [0.25, 0.5, 0.75]))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid continuous marginal for {key}.") from exc
            if not valid:
                raise ValueError(f"Invalid continuous marginal for {key}.")
            self._marginals[key] = distribution

        self.correlation_zx = self._parameter(correlation_zx, (self.k,), "correlation_zx")
        self.correlation_yx = self._parameter(correlation_yx, (self.ky,), "correlation_yx")
        self.conditional_correlation_zy = self._parameter(
            conditional_correlation_zy, (self.k, self.ky), "conditional_correlation_zy")
        residual = np.block([[np.eye(self.k), self.conditional_correlation_zy],
                             [self.conditional_correlation_zy.T, np.eye(self.ky)]])
        try:
            self._residual_cholesky = np.linalg.cholesky(residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Conditional residual correlation must be positive definite.") from exc
        self._loading = np.r_[self.correlation_zx, self.correlation_yx]
        self._scale = np.sqrt(1 - self._loading**2)
        covariance = residual * np.outer(self._scale, self._scale)
        correlation = covariance + np.outer(self._loading, self._loading)
        self.correlation_matrix = np.block([[correlation, self._loading[:, None]],
                                            [self._loading[None, :], np.ones((1, 1))]])
        try:
            np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Joint copula correlation must be positive definite.") from exc
        self.is_null = bool(np.all(self.conditional_correlation_zy == 0))
        self.Z = self.Y = self.X = None

    @staticmethod
    def _parameter(value, shape, name):
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            value = np.full(shape, value.item())
        if value.shape != shape or not np.isfinite(value).all() or np.any(np.abs(value) >= 1):
            raise ValueError(f"{name} must be finite, strictly between -1 and 1, and scalar or shape {shape}.")
        return value.copy()

    def _transform(self, key, normal):
        # Keep inverse CDF inputs inside the open unit interval in extreme tails.
        uniform = np.clip(stats.norm.cdf(normal), np.nextafter(0., 1.), np.nextafter(1., 0.))
        return self._marginals[key].ppf(uniform)

    def sample_latent(self):
        ux = self.rng.standard_normal((self.n, 1))
        residual = self.rng.standard_normal((self.n, self.k + self.ky)) @ self._residual_cholesky.T
        joint = ux * self._loading + residual * self._scale
        self.Z = self._transform("z", joint[:, :self.k])
        self.Y = self._transform("y", joint[:, self.k:])
        self.X = self._transform("x", ux)
        return self.Z, self.Y, self.X

    def sample(self, X=None):
        if X is None:
            return self.sample_latent()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[1] != 1 or not np.isfinite(X).all():
            raise ValueError("X must contain finite values with shape (n,) or (n, 1).")
        uniform = self._marginals["x"].cdf(X)
        if not np.isfinite(uniform).all() or np.any((uniform <= 0) | (uniform >= 1)):
            raise ValueError("X must lie inside the numerically invertible support of its marginal.")
        ux = stats.norm.ppf(uniform)
        uy = (ux * self.correlation_yx + self.rng.standard_normal((len(X), self.ky))
              * np.sqrt(1 - self.correlation_yx**2))
        return self._transform("y", uy)

    def get_name(self):
        return (f"continuous_conditional_copula_gaussian_zx{self.correlation_zx.tolist()}_"
                f"yx{self.correlation_yx.tolist()}_zy{self.conditional_correlation_zy.tolist()}")
