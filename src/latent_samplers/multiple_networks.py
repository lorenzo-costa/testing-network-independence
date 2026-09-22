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


class _StandardizedStudentT:
    """Independent Student-t entries standardized to unit population variance."""

    def __init__(self, df):
        self.df = df
        self._standard_deviation = np.sqrt(df / (df - 2))

    def rvs(self, *, loc, scale, size, random_state):
        draws = stats.t.rvs(df=self.df, size=size, random_state=random_state)
        return np.asarray(loc) + np.asarray(scale) * draws / self._standard_deviation


_STUDENT_T_3 = _StandardizedStudentT(df=3)


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
    B : {None, 0} or array-like of shape (d_y, p * d_x), optional
        Fixed coefficient matrix. Scalar 0 creates an all-zero matrix. If None
        or omitted, a new matrix is drawn on each call to :meth:`sample_latent`.
    snr : float, optional
        Nonnegative population variance ratio, conditional on B:
        ``trace(B @ x_covariance @ B.T) / trace(eps_covariance)``.
        Rescale supplied or sampled B to attain this ratio, leaving the error
        covariance unchanged. None preserves the original coefficients; zero
        sets them to zero. Requires positive noise variance, and positive
        signal variance before scaling when snr > 0. Returned B is rescaled.
    x_mean : float or array-like of shape (p * d_x,), default=0
        Mean of the concatenated X positions, in network block order.
    x_variance : float or array-like of shape (p * d_x, p * d_x), default=1
        Nonnegative variance, expanded as variance times identity, or a full
        positive-semidefinite covariance, including cross-network covariance.
    x_network_correlation : float, optional
        Equicorrelation between matching dimensions in different X networks.
        Different latent dimensions remain independent. Requires scalar
        ``x_variance``. None preserves the covariance specified by
        ``x_variance`` directly.
    eps_variance : float or array-like of shape (d_y, d_y), default=1
        Variance or positive-semidefinite covariance of zero-mean errors.
    b_mean, b_variance : float, default=0, 1
        Mean and nonnegative variance of independent coefficient entries.
    b_active_network_fraction : float, optional
        Fraction of X-network coefficient blocks to keep nonzero when B is
        sampled. On every call to :meth:`sample_latent`, exactly
        ``round(fraction * p)`` networks are selected uniformly without
        replacement; all coefficients in the other network blocks are zero.
        Half-integers round up. None leaves every sampled block active.
    x_distribution, eps_distribution : str, default="multivariate_gaussian"
        Names in :attr:`distribution_registry`. Full covariances require a
        distribution registered with kind ``"multivariate"``. The
        ``"student_t_3"`` distribution draws independent, variance-standardized
        univariate t errors with three degrees of freedom.
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
    SNR calibration uses the configured covariances, not sample variances or
    squared means. Registered X/error distributions must honor those moments
    for the population SNR guarantee to hold.
    """

    distribution_registry = {
        "gaussian": stats.norm,
        "multivariate_gaussian": stats.multivariate_normal,
        "student_t_3": _STUDENT_T_3,
    }
    _distribution_kinds = {
        "gaussian": "univariate",
        "multivariate_gaussian": "multivariate",
        "student_t_3": "univariate",
    }

    def __init__(
        self,
        n,
        p,
        d_x,
        d_y,
        *,
        B=None,
        snr=None,
        x_mean=0,
        x_variance=1,
        x_network_correlation=None,
        eps_variance=1,
        b_mean=0,
        b_variance=1,
        b_active_network_fraction=None,
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
        self.x_network_correlation = self._network_correlation(
            x_network_correlation
        )
        x_variance = self._network_covariance(x_variance)
        self.x_covariance = self._normalize_covariance(
            x_variance, width, "x_variance", self._x_distribution[1]
        )
        self.eps_covariance = self._normalize_covariance(
            eps_variance, self.d_y, "eps_variance", self._eps_distribution[1]
        )
        self.b_mean = _finite_scalar(b_mean, "b_mean")
        self.b_variance = _finite_scalar(b_variance, "b_variance", nonnegative=True)
        if isinstance(b_active_network_fraction, (bool, np.bool_)):
            raise ValueError(
                "b_active_network_fraction must be a finite scalar in [0, 1] "
                "or None."
            )
        self.b_active_network_fraction = (
            None
            if b_active_network_fraction is None
            else _finite_scalar(
                b_active_network_fraction, "b_active_network_fraction"
            )
        )
        if (
            self.b_active_network_fraction is not None
            and not 0 <= self.b_active_network_fraction <= 1
        ):
            raise ValueError("b_active_network_fraction must be in [0, 1].")
        if isinstance(snr, (bool, np.bool_)):
            raise ValueError("snr must be a nonnegative finite scalar or None.")
        self.snr = None if snr is None else _finite_scalar(snr, "snr", nonnegative=True)
        self._noise_power = None
        if self.snr is not None:
            with np.errstate(over="ignore"):
                self._noise_power = float(np.trace(self.eps_covariance))
            if not np.isfinite(self._noise_power) or self._noise_power <= 0:
                raise ValueError(
                    "snr requires a positive, finite trace of eps_variance."
                )
        self.B = None
        if B is not None:
            self.B = _finite_array(B, "B")
            if self.B.ndim == 0 and np.asarray(B).dtype.kind in "iuf" and self.B == 0:
                self.B = np.zeros((self.d_y, width))
            if self.B.shape != (self.d_y, width):
                raise ValueError(
                    f"B must be 0, None, or have shape ({self.d_y}, {width}); "
                    f"got {self.B.shape}."
                )
            self.B = self._rescale_B(self.B)

    def _rescale_B(self, B):
        """Calibrate coefficient magnitude without drawing randomness."""
        if self.snr is None:
            return B
        if self.snr == 0:
            return np.zeros_like(B)
        with np.errstate(over="ignore", invalid="ignore"):
            signal_power = float(np.sum((B @ self.x_covariance) * B))
        if not np.isfinite(signal_power) or signal_power <= 0:
            raise ValueError(
                "Positive snr requires positive, finite signal variance "
                "trace(B @ x_covariance @ B.T); B=0 or degenerate X covariance "
                "cannot supply a positive signal."
            )
        # Work in log scale to avoid overflowing snr * noise_power before division.
        log_scale = 0.5 * (
            np.log(self.snr) + np.log(self._noise_power) - np.log(signal_power)
        )
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scaled_B = B * np.exp(log_scale)
        if not np.isfinite(scaled_B).all() or not np.any(scaled_B):
            raise ValueError("snr scaling cannot produce finite, nonzero coefficients.")
        return scaled_B

    def _select_active_networks(self, B):
        """Zero coefficient blocks outside a newly sampled uniform subset."""
        if self.b_active_network_fraction is None:
            return B
        active_count = int(np.floor(self.b_active_network_fraction * self.p + 0.5))
        if active_count == self.p:
            return B
        sparse_B = np.zeros_like(B)
        if active_count:
            active = self.rng.choice(self.p, size=active_count, replace=False)
            for network in active:
                start = network * self.d_x
                sparse_B[:, start : start + self.d_x] = B[:, start : start + self.d_x]
        return sparse_B

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

    def _network_correlation(self, value):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("x_network_correlation must be a finite scalar.")
        correlation = _finite_scalar(value, "x_network_correlation")
        lower_bound = -1 if self.p == 1 else -1 / (self.p - 1)
        if correlation < lower_bound or correlation > 1:
            raise ValueError(
                "x_network_correlation must produce a positive-semidefinite "
                f"equicorrelation matrix; expected {lower_bound:g} <= rho <= 1."
            )
        return correlation

    def _network_covariance(self, x_variance):
        if self.x_network_correlation is None:
            return x_variance
        variance = _finite_array(x_variance, "x_variance")
        if variance.ndim != 0:
            raise ValueError("x_network_correlation requires scalar x_variance.")
        variance = _finite_scalar(x_variance, "x_variance", nonnegative=True)
        correlation = self.x_network_correlation
        network_correlation = (1 - correlation) * np.eye(self.p) + correlation
        return variance * np.kron(network_correlation, np.eye(self.d_x))

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
            ``(d_y, p * d_x)``. B contains the effective coefficients after
            optional SNR scaling. List order matches the column blocks of B.
            Neither errors nor concatenated X are included.
        """
        x_concat = self._draw_rows(
            self._x_distribution, self.x_mean, self.x_covariance, self.n
        )
        width = self.p * self.d_x
        B = (
            self.B.copy()
            if self.B is not None
            else self._rescale_B(
                self._select_active_networks(
                    self._draw_rows(
                        self._b_distribution,
                        np.full(width, self.b_mean),
                        self.b_variance * np.eye(width),
                        self.d_y,
                    )
                )
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
