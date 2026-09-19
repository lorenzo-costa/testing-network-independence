"""Latent positions for a linear model relating multiple networks."""

from numbers import Integral

import numpy as np
from scipy import stats


def _finite_array(value, name):
    """Convert real numerical input to an owned, finite floating-point array."""
    try:
        if np.iscomplexobj(value):
            raise ValueError
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite real numbers.") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _finite_scalar(value, name, *, nonnegative=False):
    """Validate a scalar mean or variance without accepting length-one arrays."""
    array = _finite_array(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar.")
    value = float(array)
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


class MultipleNetworksSampler:
    """Sample the row-oriented linear model ``Y = X_concat @ B.T + epsilon``.

    Parameters
    ----------
    n : int
        Positive number of independently sampled nodes.
    p : int
        Positive number of X networks.
    d_x, d_y : int
        Positive latent dimensions of each X network and the Y network.
    B : array-like of shape (d_y, p * d_x), optional
        Fixed coefficient matrix. If omitted, a new matrix is drawn on each
        call to :meth:`sample_latent`.
    x_mean : float or array-like of shape (p * d_x,), default=0
        Mean of the concatenated X positions, in network block order.
    x_variance : float or array-like of shape (p * d_x, p * d_x), default=1
        Nonnegative variance, expanded as variance times identity, or a full
        positive-semidefinite covariance, including cross-network covariance.
    eps_variance : float or array-like of shape (d_y, d_y), default=1
        Variance or positive-semidefinite covariance of zero-mean errors.
    b_mean, b_variance : float, default=0, 1
        Mean and nonnegative variance of independent coefficient entries.
    x_distribution, eps_distribution : str, default="multivariate_gaussian"
        Names in :attr:`distribution_registry`. Full covariances require a
        distribution registered with kind ``"multivariate"``.
    b_distribution : str, default="gaussian"
        Registered distribution for coefficient entries.
    rng : numpy.random.Generator, optional
        Shared source of all randomness. Created with ``default_rng`` if absent.

    Notes
    -----
    The public registry maps names directly to SciPy distribution objects.
    Calling conventions are stored separately and resolved once at construction.
    Univariate distributions draw independent entries using ``loc`` and
    ``scale=sqrt(variance)``; multivariate distributions draw independent rows
    using ``mean`` and ``cov``. No global random state is used.
    """

    distribution_registry = {
        "gaussian": stats.norm,
        "multivariate_gaussian": stats.multivariate_normal,
    }
    _distribution_kinds = {
        "gaussian": "univariate",
        "multivariate_gaussian": "multivariate",
    }

    def __init__(
        self,
        n,
        p,
        d_x,
        d_y,
        *,
        B=None,
        x_mean=0,
        x_variance=1,
        eps_variance=1,
        b_mean=0,
        b_variance=1,
        x_distribution="multivariate_gaussian",
        eps_distribution="multivariate_gaussian",
        b_distribution="gaussian",
        rng=None,
    ):
        self.n = self._positive_integer(n, "n")
        self.p = self._positive_integer(p, "p")
        self.d_x = self._positive_integer(d_x, "d_x")
        self.d_y = self._positive_integer(d_y, "d_y")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator or None.")
        self.rng = rng if rng is not None else np.random.default_rng()

        self.x_distribution = x_distribution
        self.eps_distribution = eps_distribution
        self.b_distribution = b_distribution
        self._x_distribution = self._resolve_distribution(
            x_distribution, "x_distribution"
        )
        self._eps_distribution = self._resolve_distribution(
            eps_distribution, "eps_distribution"
        )
        self._b_distribution = self._resolve_distribution(
            b_distribution, "b_distribution"
        )

        width = self.p * self.d_x
        self.x_mean = self._normalize_mean(x_mean, width, "x_mean")
        self.x_covariance = self._normalize_covariance(
            x_variance, width, "x_variance", self._x_distribution[1]
        )
        self.eps_covariance = self._normalize_covariance(
            eps_variance, self.d_y, "eps_variance", self._eps_distribution[1]
        )
        self.b_mean = _finite_scalar(b_mean, "b_mean")
        self.b_variance = _finite_scalar(b_variance, "b_variance", nonnegative=True)
        self.B = None
        if B is not None:
            self.B = _finite_array(B, "B")
            if self.B.shape != (self.d_y, width):
                raise ValueError(
                    f"B must have shape ({self.d_y}, {width}); got {self.B.shape}."
                )

    @classmethod
    def register_distribution(cls, name, distribution, *, kind="univariate"):
        """Register a distribution with a compatible SciPy ``rvs`` interface.

        Parameters
        ----------
        name : str
            Nonempty lookup name shared by all three distribution parameters.
        distribution : object
            Object exposing ``rvs`` with ``random_state`` and either
            ``loc, scale, size`` or ``mean, cov, size`` arguments. Distributions
            needing additional shape parameters must bind them in an adapter.
        kind : {"univariate", "multivariate"}, default="univariate"
            Calling convention for drawing independent entries or rows.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Distribution name must be a nonempty string.")
        if kind not in {"univariate", "multivariate"}:
            raise ValueError(
                "Distribution kind must be 'univariate' or 'multivariate'."
            )
        if not callable(getattr(distribution, "rvs", None)):
            raise ValueError("Distribution must provide a callable rvs method.")
        # Copy on registration so subclass extensions do not modify the parent.
        cls.distribution_registry = {**cls.distribution_registry, name: distribution}
        cls._distribution_kinds = {**cls._distribution_kinds, name: kind}

    @classmethod
    def _resolve_distribution(cls, name, parameter):
        if not isinstance(name, str) or name not in cls.distribution_registry:
            available = ", ".join(sorted(cls.distribution_registry))
            raise ValueError(
                f"Unknown {parameter}={name!r}. Available names: {available}."
            )
        return cls.distribution_registry[name], cls._distribution_kinds[name]

    @staticmethod
    def _positive_integer(value, name):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    @staticmethod
    def _normalize_mean(value, dimension, name):
        mean = _finite_array(value, name)
        if mean.ndim == 0:
            return np.full(dimension, float(mean))
        if mean.shape != (dimension,):
            raise ValueError(
                f"{name} must be scalar or have shape ({dimension},); got {mean.shape}."
            )
        return mean

    @staticmethod
    def _normalize_covariance(value, dimension, name, kind):
        covariance = _finite_array(value, name)
        if covariance.ndim == 0:
            variance = _finite_scalar(value, name, nonnegative=True)
            return variance * np.eye(dimension)
        if covariance.shape != (dimension, dimension):
            raise ValueError(
                f"{name} must be scalar or have shape ({dimension}, {dimension}); "
                f"got {covariance.shape}."
            )
        if kind != "multivariate":
            raise ValueError(
                f"{name} is a full covariance matrix; choose a multivariate distribution."
            )
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-10):
            raise ValueError(f"{name} must be symmetric.")
        covariance = 0.5 * covariance + 0.5 * covariance.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = 1e-10 * max(1.0, np.max(np.abs(eigenvalues)))
        if eigenvalues[0] < -tolerance:
            raise ValueError(f"{name} must be positive semidefinite.")
        if eigenvalues[0] < 0:
            # Remove only roundoff-sized negative eigenvalues, on an owned copy.
            covariance = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        return covariance

    def _draw_rows(self, resolved_distribution, mean, covariance, count):
        """Draw independent rows and restore axes squeezed by SciPy."""
        distribution, kind = resolved_distribution
        if kind == "multivariate":
            values = distribution.rvs(
                mean=mean, cov=covariance, size=count, random_state=self.rng
            )
        else:
            values = distribution.rvs(
                loc=mean,
                scale=np.sqrt(np.diag(covariance)),
                size=(count, len(mean)),
                random_state=self.rng,
            )
        return np.asarray(values, dtype=float).reshape(count, len(mean))

    def sample_latent(self):
        """Draw X positions, coefficients (unless fixed), and independent errors.

        Returns
        -------
        dict
            Exactly ``Y``, an array of shape ``(n, d_y)``; ``X``, a Python list
            of ``p`` arrays of shape ``(n, d_x)``; and ``B``, an array of shape
            ``(d_y, p * d_x)``. List order matches the column blocks of B.
            Neither errors nor concatenated X are included.
        """
        x_concat = self._draw_rows(
            self._x_distribution, self.x_mean, self.x_covariance, self.n
        )
        width = self.p * self.d_x
        B = (
            self.B.copy()
            if self.B is not None
            else self._draw_rows(
                self._b_distribution,
                np.full(width, self.b_mean),
                self.b_variance * np.eye(width),
                self.d_y,
            )
        )
        epsilon = self._draw_rows(
            self._eps_distribution, np.zeros(self.d_y), self.eps_covariance, self.n
        )
        return {
            "Y": x_concat @ B.T + epsilon,
            "X": list(np.split(x_concat, self.p, axis=1)),
            "B": B,
        }
