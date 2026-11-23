"""Modulo-one conditional sampler."""

from numbers import Integral

import numpy as np


class ConditionalSampler:
    """Sample scalar X independently of Z, then set Y = (Z + X) mod 1.

    For ky=1, Y uses the first Z coordinate. For ky=k, each Y coordinate
    uses its matching Z coordinate and the same scalar X. Uniform Z makes
    Y uniform conditional on X, regardless of the distribution of X.
    """

    def __init__(self, n, k, ky=1, *, z_distribution="uniform",
                 x_distribution="uniform", column_covariance=None,
                 noise_scale=0.0, rng=None):
        for name, value in (("n", n), ("k", k), ("ky", ky)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if ky not in (1, k):
            raise ValueError("ky must equal 1 or k.")
        if z_distribution not in ("uniform", "gaussian", "cauchy", "uniform_rdpg"):
            raise ValueError("Unsupported z_distribution.")
        if x_distribution not in ("uniform", "gaussian", "cauchy"):
            raise ValueError("Unsupported x_distribution.")
        if not np.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative.")
        self.n, self.kz, self.ky = n, k, ky
        self.z_distribution = z_distribution
        self.x_distribution = x_distribution
        self.noise_scale = float(noise_scale)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.column_covariance = (np.eye(k) if column_covariance is None
                                  else np.asarray(column_covariance, dtype=float))
        if self.column_covariance.shape != (k, k) or not np.all(np.isfinite(self.column_covariance)):
            raise ValueError("column_covariance must be a finite (k, k) matrix.")
        if not np.allclose(self.column_covariance, self.column_covariance.T):
            raise ValueError("column_covariance must be symmetric.")
        if np.linalg.eigvalsh(self.column_covariance).min() < -1e-10:
            raise ValueError("column_covariance must be positive semidefinite.")
        self.is_null = False

    def _sample_z(self):
        shape = (self.n, self.kz)
        if self.z_distribution == "uniform":
            return self.rng.uniform(size=shape)
        if self.z_distribution == "gaussian":
            return self.rng.multivariate_normal(np.zeros(self.kz), self.column_covariance, size=self.n)
        if self.z_distribution == "cauchy":
            return self.rng.standard_cauchy(size=shape)
        z = np.abs(self.rng.normal(size=shape))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z * self.rng.uniform(size=(self.n, 1)) ** (1 / self.kz)

    def _sample_x(self):
        shape = (self.n, 1)
        if self.x_distribution == "uniform":
            return self.rng.uniform(size=shape)
        if self.x_distribution == "gaussian":
            return self.rng.normal(size=shape)
        return self.rng.standard_cauchy(size=shape)

    def sample(self, x=None):
        """Return (Z, Y, X), or (Z, Y) when X is supplied."""
        supplied = x is not None
        if supplied:
            x = np.asarray(x, dtype=float)
            if x.shape == (self.n,):
                x = x[:, None]
            if x.shape != (self.n, 1) or not np.all(np.isfinite(x)):
                raise ValueError(f"x must have finite shape ({self.n}, 1).")
        else:
            x = self._sample_x()
        z = self._sample_z()
        y = np.mod(z[:, :self.ky] + x, 1.0)
        if self.noise_scale:
            y += self.rng.normal(scale=self.noise_scale, size=y.shape)
        return (z, y) if supplied else (z, y, x)

    def sample_latent(self):
        return self.sample()

    def _sample_latent(self):
        return self.sample_latent()

    def get_name(self):
        return (f"conditional_mod1_n{self.n}_k{self.kz}_ky{self.ky}_"
                f"z{self.z_distribution}_x{self.x_distribution}_noise{self.noise_scale:g}")
