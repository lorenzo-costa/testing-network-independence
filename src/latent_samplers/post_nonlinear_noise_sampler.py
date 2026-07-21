"""Post-nonlinear-noise sampler with a categorical conditioning variable."""

from numbers import Integral, Real
import warnings

import numpy as np


class PostNonLinearNoiseSampler:
    """Sample ``(Z, Y, X)`` from a stratified post-nonlinear-noise model.

    The categorical variable ``X`` is sampled first.  Each category has an
    independently drawn joint effect ``(a_c, b_c)``.  Gaussian errors are then
    added and a coordinate-wise nonlinear function is applied to each output
    coordinate.  ``function_type="random"`` selects functions independently
    by coordinate; a named function applies the same transformation to every
    coordinate.  The selected functions are fixed for the complete generated
    data set.

    Parameters
    ----------
    n : int
        Number of observations.
    k : int
        Number of coordinates in ``Z``.
    ky : int, default=1
        Number of coordinates in ``Y``.
    C : int
        Number of categorical strata.
    rho : float or array-like, optional
        Error correlation.  A scalar is shared by all strata; a vector must
        contain one correlation per stratum.  Ignored when
        ``error_covariance`` is supplied.
    error_covariance : array-like, optional
        Full ``(k + ky)``-dimensional covariance of ``(E_Z, E_Y)``.  When
        omitted, the covariance is built from ``rho`` and the two column
        covariance matrices.
    stratum_covariance : array-like, optional
        Covariance of each independently drawn joint category effect
        ``(a_c, b_c)``.  Defaults to identity.
    column_covariance_z, column_covariance_y : array-like, optional
        Within-block error covariances used when constructing the error
        covariance from ``rho``.  Both default to identity.
    cross_correlation_template : array-like, optional
        ``(k, ky)`` pairing matrix used in the cross-covariance construction.
        Defaults to a rectangular identity.
    class_probabilities : array-like, optional
        Probabilities of the ``C`` strata.  Defaults to uniform.
    p : array-like, optional
        Alias for ``class_probabilities``.
    center_latent : bool, default=True
        Center every coordinate of ``Z`` and ``Y`` within each sampled stratum
        after applying the nonlinear functions.
    function_type : {"random", "identity", "square", "tanh", "exp_neg_square"}, default="random"
        Nonlinear transformation applied to ``Z`` and ``Y``.  ``"random"``
        samples one of the four named transformations independently for each
        coordinate.  Any other accepted value applies that transformation to
        every coordinate.
    rng : numpy.random.Generator, optional
        Random number generator.
    """

    _FUNCTION_NAMES = ("identity", "square", "tanh", "exp_neg_square")

    def __init__(
        self,
        n,
        k,
        C,
        ky=1,
        rho=None,
        error_covariance=None,
        stratum_covariance=None,
        column_covariance_z=None,
        column_covariance_y=None,
        column_covariance=None,
        cross_correlation_template=None,
        class_probabilities=None,
        p=None,
        center_latent=True,
        function_type="random",
        rng=None,
        **kwargs,
    ):
        # LatentSampler receives shared configuration dictionaries.  Preserve
        # irrelevant fields for inspection, but deliberately do not let them
        # prevent this sampler from being selected.
        self.unused_kwargs = dict(kwargs)

        self.n = self._positive_integer(n, "n")
        self.k = self._positive_integer(k, "k")
        self.ky = self._positive_integer(ky, "ky")
        self.C = self._positive_integer(C, "C")
        self.dimension = self.k + self.ky
        self.rng = rng if rng is not None else np.random.default_rng()
        self.center_latent = bool(center_latent)
        self.function_type = self._normalize_function_type(function_type)

        if class_probabilities is not None and p is not None:
            raise ValueError("Specify only one of class_probabilities or p.")
        self.class_probabilities = self._normalize_probabilities(
            class_probabilities if class_probabilities is not None else p
        )

        if column_covariance is not None and column_covariance_z is not None:
            raise ValueError(
                "Specify only one of column_covariance or column_covariance_z."
            )
        if column_covariance_z is None:
            column_covariance_z = column_covariance
        self.column_covariance_z = self._positive_definite_covariance(
            np.eye(self.k) if column_covariance_z is None else column_covariance_z,
            self.k,
            "column_covariance_z",
        )
        self.column_covariance_y = self._positive_definite_covariance(
            np.eye(self.ky) if column_covariance_y is None else column_covariance_y,
            self.ky,
            "column_covariance_y",
        )
        self.cross_correlation_template = self._cross_template(
            cross_correlation_template
        )
        self.stratum_covariance = self._covariance(
            np.eye(self.dimension)
            if stratum_covariance is None
            else stratum_covariance,
            self.dimension,
            "stratum_covariance",
        )

        self.rho = self._normalize_rho(0 if rho is None else rho)
        if error_covariance is not None:
            if rho is not None:
                warnings.warn(
                    "Both error_covariance and rho were supplied; "
                    "error_covariance takes precedence.",
                    UserWarning,
                    stacklevel=2,
                )
            covariance = self._covariance(
                error_covariance,
                self.dimension,
                "error_covariance",
            )
            self.error_covariances = np.repeat(
                covariance[None, :, :], self.C, axis=0
            )
            self.error_covariance = covariance
        else:
            self.error_covariances = np.stack(
                [self._build_error_covariance(value) for value in self.rho]
            )
            self.error_covariance = None

        cross_blocks = self.error_covariances[:, : self.k, self.k :]
        self.is_null = bool(np.allclose(cross_blocks, 0.0))
        self.Z = None
        self.Y = None
        self.X = None
        self.stratum_effects = None
        self.a = None
        self.b = None
        self.nonlinear_functions_z = None
        self.nonlinear_functions_y = None

    @staticmethod
    def _positive_integer(value, name):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    @classmethod
    def _normalize_function_type(cls, function_type):
        if not isinstance(function_type, str):
            raise TypeError("function_type must be a string.")
        function_type = function_type.strip().lower()
        choices = ("random",) + cls._FUNCTION_NAMES
        if function_type not in choices:
            available = ", ".join(choices)
            raise ValueError(
                f"Unknown function_type {function_type!r}. Available: {available}."
            )
        return function_type

    def _normalize_probabilities(self, probabilities):
        if probabilities is None:
            return np.full(self.C, 1.0 / self.C)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 1 or probabilities.size != self.C:
            raise ValueError(
                f"class_probabilities must be a vector of length C={self.C}."
            )
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError(
                "class_probabilities must contain finite, nonnegative values."
            )
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("class_probabilities must sum to 1.")
        return probabilities

    def _normalize_rho(self, rho):
        if isinstance(rho, Real) and not isinstance(rho, bool):
            values = np.full(self.C, float(rho))
        else:
            values = np.asarray(rho, dtype=float)
            if values.ndim != 1 or values.size != self.C:
                raise ValueError(f"rho must be a scalar or a vector of length C={self.C}.")
        if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
            raise ValueError("rho values must lie in [-1, 1].")
        return values

    @staticmethod
    def _covariance(value, dimension, name):
        covariance = np.asarray(value, dtype=float)
        if covariance.shape != (dimension, dimension):
            raise ValueError(f"{name} must have shape ({dimension}, {dimension}).")
        if not np.isfinite(covariance).all():
            raise ValueError(f"{name} must contain only finite values.")
        if not np.allclose(covariance, covariance.T):
            raise ValueError(f"{name} must be symmetric.")
        if np.linalg.eigvalsh(covariance).min() < -1e-10:
            raise ValueError(f"{name} must be positive semidefinite.")
        return covariance

    @classmethod
    def _positive_definite_covariance(cls, value, dimension, name):
        covariance = cls._covariance(value, dimension, name)
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be positive definite.") from exc
        return covariance

    def _cross_template(self, template):
        if template is None:
            template = np.zeros((self.k, self.ky))
            diagonal = np.arange(min(self.k, self.ky))
            template[diagonal, diagonal] = 1.0
        template = np.asarray(template, dtype=float)
        if template.shape != (self.k, self.ky):
            raise ValueError(
                "cross_correlation_template must have shape "
                f"({self.k}, {self.ky})."
            )
        if not np.isfinite(template).all():
            raise ValueError(
                "cross_correlation_template must contain only finite values."
            )
        return template

    def _build_error_covariance(self, rho):
        Lz = np.linalg.cholesky(self.column_covariance_z)
        Ly = np.linalg.cholesky(self.column_covariance_y)
        cross = rho * Lz @ self.cross_correlation_template @ Ly.T
        covariance = np.block(
            [
                [self.column_covariance_z, cross],
                [cross.T, self.column_covariance_y],
            ]
        )
        return self._covariance(covariance, self.dimension, "constructed error covariance")

    @staticmethod
    def _apply_function(values, name):
        if name == "identity":
            return values
        if name == "square":
            return values**2
        if name == "tanh":
            return np.tanh(values)
        if name == "exp_neg_square":
            return np.exp(-(values**2))
        raise RuntimeError(f"Unknown nonlinear function {name!r}.")

    def _apply_random_functions(self, values):
        names = self.rng.choice(self._FUNCTION_NAMES, size=values.shape[1])
        return self._apply_named_functions(values, names)

    def _apply_named_functions(self, values, names):
        transformed = np.empty_like(values, dtype=float)
        for coordinate, name in enumerate(names):
            transformed[:, coordinate] = self._apply_function(
                values[:, coordinate], name
            )
        return transformed, np.asarray(names)

    def _apply_functions(self, values):
        if self.function_type == "random":
            return self._apply_random_functions(values)
        names = np.full(values.shape[1], self.function_type)
        return self._apply_named_functions(values, names)

    def sample_latent(self):
        """Return ``Z``, ``Y``, and categorical ``X`` with shape ``(n, 1)``."""
        labels = self.rng.choice(
            self.C,
            size=self.n,
            replace=True,
            p=self.class_probabilities,
        )
        effects = self.rng.multivariate_normal(
            mean=np.zeros(self.dimension),
            cov=self.stratum_covariance,
            size=self.C,
            check_valid="raise",
        )
        raw = np.empty((self.n, self.dimension), dtype=float)
        for stratum in range(self.C):
            mask = labels == stratum
            size = int(mask.sum())
            if size == 0:
                continue
            errors = self.rng.multivariate_normal(
                mean=np.zeros(self.dimension),
                cov=self.error_covariances[stratum],
                size=size,
                check_valid="raise",
            )
            raw[mask] = effects[stratum] + errors

        Z, functions_z = self._apply_functions(raw[:, : self.k])
        Y, functions_y = self._apply_functions(raw[:, self.k :])

        if self.center_latent:
            for stratum in range(self.C):
                mask = labels == stratum
                if np.any(mask):
                    Z[mask] -= Z[mask].mean(axis=0, keepdims=True)
                    Y[mask] -= Y[mask].mean(axis=0, keepdims=True)

        X = labels.reshape(self.n, 1)
        self.Z = Z
        self.Y = Y
        self.X = X
        self.stratum_effects = effects
        self.a = effects[:, : self.k]
        self.b = effects[:, self.k :]
        self.nonlinear_functions_z = functions_z
        self.nonlinear_functions_y = functions_y
        return Z, Y, X

    def _sample_latent_post_nonlinear_noise(self):
        return self.sample_latent()

    def _sample_latent(self):
        return self.sample_latent()

    def sample(self):
        return self.sample_latent()

    def get_name(self):
        return f"post_nonlinear_noise_C{self.C}_rho{self.rho.tolist()}"
