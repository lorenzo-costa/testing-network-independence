"""Categorical Gaussian data with an explicit conditional response law."""

from numbers import Integral

import numpy as np
from scipy.stats import multivariate_normal


class ConditionalGaussianSampler:
    """Sample independent rows from a specified Gaussian mixture.

    ``X_i`` is categorical with probabilities ``class_probabilities``. Given
    ``X_i = c``, ``[Z_i, Y_i]`` has mean ``means[c]`` and covariance
    ``covariances[c]``. The first ``k`` coordinates belong to ``Z`` and the
    remaining ``ky`` to ``Y``. A zero cross block in every covariance gives
    conditional independence of ``Z`` and ``Y`` given ``X``.

    ``sample(X)`` draws from the exact marginal ``Y | X`` for Model-X tests.
    ``conditional_distribution(c)`` exposes that Gaussian distribution, and
    ``conditional_logpdf(Y, X)`` gives one log density per row.
    """

    def __init__(
        self,
        n,
        k,
        ky=1,
        *,
        means,
        covariances,
        class_probabilities,
        rng=None,
    ):
        self.n = self._positive_integer(n, "n")
        self.k = self._positive_integer(k, "k")
        self.ky = self._positive_integer(ky, "ky")
        self.rng = rng if rng is not None else np.random.default_rng()

        probabilities = np.asarray(class_probabilities, dtype=float)
        if probabilities.ndim != 1 or probabilities.size < 1:
            raise ValueError("class_probabilities must be a nonempty vector.")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("class_probabilities must be finite and nonnegative.")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0):
            raise ValueError("class_probabilities must sum to 1.")
        self.class_probabilities = probabilities / probabilities.sum()
        self.C = probabilities.size

        dimension = self.k + self.ky
        means = np.asarray(means, dtype=float)
        covariances = np.asarray(covariances, dtype=float)
        if means.shape != (self.C, dimension):
            raise ValueError(f"means must have shape ({self.C}, {dimension}).")
        if covariances.shape != (self.C, dimension, dimension):
            raise ValueError(
                f"covariances must have shape ({self.C}, {dimension}, {dimension})."
            )
        if not np.isfinite(means).all() or not np.isfinite(covariances).all():
            raise ValueError("means and covariances must contain only finite values.")
        if not np.allclose(covariances, covariances.swapaxes(-1, -2), atol=1e-12, rtol=0):
            raise ValueError("Each covariance must be symmetric.")
        covariances = (covariances + covariances.swapaxes(-1, -2)) / 2
        try:
            joint_cholesky = np.linalg.cholesky(covariances)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Each covariance must be positive definite.") from exc

        self.means = means.copy()
        self.covariances = covariances.copy()
        self._joint_cholesky = joint_cholesky
        self._y_cholesky = np.linalg.cholesky(self.covariances[:, self.k:, self.k:])
        self.is_null = bool(np.all(self.covariances[:, :self.k, self.k:] == 0))
        self.Z = None
        self.Y = None
        self.X = None

    @staticmethod
    def _positive_integer(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _labels(self, X):
        labels = np.asarray(X)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1:
            raise ValueError("X must have shape (n,) or (n, 1).")
        if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(
            labels.dtype, np.bool_
        ):
            raise ValueError("X must contain integer category labels.")
        if np.any((labels < 0) | (labels >= self.C)):
            raise ValueError(f"X labels must lie in 0, ..., {self.C - 1}.")
        return labels

    def sample_latent(self):
        """Draw and retain ``(Z, Y, X)`` with shapes ``(n,k)``, ``(n,ky)``, ``(n,1)``."""
        labels = self.rng.choice(self.C, size=self.n, p=self.class_probabilities)
        joint = np.empty((self.n, self.k + self.ky))
        for category in range(self.C):
            mask = labels == category
            count = int(mask.sum())
            if count:
                joint[mask] = (
                    self.rng.standard_normal((count, self.k + self.ky))
                    @ self._joint_cholesky[category].T
                    + self.means[category]
                )
        self.Z = joint[:, :self.k]
        self.Y = joint[:, self.k:]
        self.X = labels[:, None]
        return self.Z, self.Y, self.X

    def sample(self, X):
        """Draw independent response rows from ``p(Y_i | X_i)``."""
        labels = self._labels(X)
        response = np.empty((labels.size, self.ky))
        for category in range(self.C):
            mask = labels == category
            count = int(mask.sum())
            if count:
                response[mask] = (
                    self.rng.standard_normal((count, self.ky))
                    @ self._y_cholesky[category].T
                    + self.means[category, self.k:]
                )
        return response

    def conditional_distribution(self, category):
        """Return the frozen SciPy distribution of ``Y | X=category``."""
        if (
            isinstance(category, bool)
            or not isinstance(category, Integral)
            or not 0 <= category < self.C
        ):
            raise ValueError(f"category must be an integer in 0, ..., {self.C - 1}.")
        return multivariate_normal(
            mean=self.means[category, self.k:],
            cov=self.covariances[category, self.k:, self.k:],
        )

    def conditional_logpdf(self, Y, X):
        """Return the log density of each row of ``Y`` conditional on ``X``."""
        labels = self._labels(X)
        response = np.asarray(Y, dtype=float)
        if response.shape != (labels.size, self.ky):
            raise ValueError(f"Y must have shape ({labels.size}, {self.ky}).")
        if not np.isfinite(response).all():
            raise ValueError("Y must contain only finite values.")
        result = np.empty(labels.size)
        for category in range(self.C):
            mask = labels == category
            if not mask.any():
                continue
            centered = response[mask] - self.means[category, self.k:]
            whitened = np.linalg.solve(self._y_cholesky[category], centered.T)
            log_determinant = np.log(np.diag(self._y_cholesky[category])).sum()
            result[mask] = (
                -0.5 * (self.ky * np.log(2 * np.pi) + (whitened**2).sum(axis=0))
                - log_determinant
            )
        return result

    def get_name(self):
        return f"conditional_gaussian_C{self.C}"
