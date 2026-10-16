r"""Functional data-generating processes for scalar or component-wise responses.

This module follows the output convention used by ``copula_sampler.py``:

* ``Z`` is the predictor matrix, so row ``i`` is the vector-valued predictor
  :math:`Z_i \in R^k`.
* ``Y`` is the response matrix, with one column for a scalar response or
  ``k`` columns for a component-wise response.

The default models are deterministic: ``Y[:, 0] = f(Z)``.  Hence, with
``noise_scale=0`` and ``rho=None`` they are useful alternatives for testing a dependence
statistic that should attain its functional-dependence maximum when
:math:`Y=f(Y)`.
"""

from __future__ import annotations

from copy import copy
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
from scipy.special import expit, ndtr


ArrayLike = Union[np.ndarray, Sequence[float]]
Functional = Union[str, Callable[[np.ndarray], ArrayLike]]


class FunctionalGenerator:
    """Sample vector predictors ``Z`` and a functional response ``Y``.

    Parameters
    ----------
    n : int
        Number of sampled observations.
    k : int
        Dimension of the vector predictor ``Z``.
    ky : int, default=1
        Response dimension: either 1 for a scalar response or ``k`` to apply
        the functional separately to each predictor column.  In component-wise
        mode, parameters follow the one-dimensional contract and index
        normalization uses unit variance for every column.  ``interaction``
        and ``hierarchical`` cannot be used in this mode when ``k > 1``.
    functional : str or callable, default="linear"
        Functional relationship used to construct the response.  Supported
        strings are ``linear``, ``nonlinear_additive``, ``interaction``,
        ``radial``, ``threshold``, ``sigmoid``, ``piecewise``, ``oscillatory``,
        ``max``, ``abs_max``, ``tanh_product``, ``sparse_nonlinear``,
        ``hierarchical``, ``manifold``, and ``pareto_index``.  A callable must
        accept an ``(n, k)`` array and return shape ``(n,)`` or ``(n, 1)``.
        In component-wise mode it is called once per column with shape ``(n, 1)``.
    function_params : dict, optional
        Parameters specific to the selected functional.  Useful keys include
        ``weights``, ``frequency``, ``threshold``, ``active_features``,
        ``alpha`` and ``scale`` (for ``pareto_index``), ``switch_feature``
        (for ``piecewise``), ``output_center`` and ``output_sd`` (for
        ``interaction``), and ``center`` and ``input_scale`` (for ``abs_max``).
    column_covariance : ndarray, optional
        ``k y k`` covariance matrix used for Gaussian predictors.  The
        identity is used by default.
    predictor_distribution : {"gaussian", "student_t", "uniform_rdpg"}, default="gaussian"
        Distribution used for the predictor vector.  For ``student_t``,
        ``predictor_df`` controls the degrees of freedom and
        ``column_covariance`` is the scale matrix. ``uniform_rdpg`` samples
        uniformly from the positive orthant of the unit Euclidean ball.  It is
        intended for ``BernoulliNetwork(rdpg=True)``: each sampled row has norm
        at most one and nonnegative entries, so every pairwise inner product is
        in ``[0, 1]``.
    predictor_df : float, default=5.0
        Degrees of freedom when ``predictor_distribution="student_t"``.
    noise_scale : float, default=0.0
        Standard deviation of response noise.  Set this to zero to preserve
        the exact deterministic relationship ``Y=f(Z)`` when ``rho=None``.
    rho : float, optional
        Target signal fraction for additive noise, with ``0 < rho <= 1``.
        Overrides ``noise_scale`` using the current sample's signal variance:
        ``noise_var = np.var(f(Z)) * (1 - rho) / rho``.  Gaussian noise is
        drawn with this target variance; its realized sample variance may
        differ.  ``rho=1`` or a constant signal produces no noise.  For a
        constant signal the variance ratio is undefined.  Ignored for
        multiplicative noise.  Component-wise responses use independent noise
        vectors and calibrate the noise variance separately for each column.
    noise_type : {"additive", "multiplicative"}, default="additive"
        Additive noise uses ``Y = f(Z) + sigma*epsilon``.  Multiplicative
        noise uses ``Y = f(Z) * exp(sigma*epsilon)`` and is especially useful
        with the heavy-tailed ``pareto_index`` model.
    center_latent : bool, default=False
        Whether to sample-center the predictor columns before applying the
        functional.  Leave this false if the exact Pareto marginal of
        ``pareto_index`` is desired.
    rng : numpy.random.Generator, optional
        Random-number generator for reproducibility.

    Notes
    -----
    ``sample_latent()`` returns ``(Z, Y)`` with shapes ``(n, k)`` and
    ``(n, ky)``.  With ``ky=1`` the functional can combine predictor columns;
    with ``ky=k > 1`` each response depends only on its corresponding column.
    """

    AVAILABLE_FUNCTIONALS = (
        "linear",
        "nonlinear_additive",
        "interaction",
        "radial",
        "threshold",
        "sigmoid",
        "piecewise",
        "oscillatory",
        "max",
        "abs_max",
        "tanh_product",
        "sparse_nonlinear",
        "hierarchical",
        "manifold",
        "pareto_index",
    )

    _ALIASES = {
        "additive": "nonlinear_additive",
        "nonlinear": "nonlinear_additive",
        "pairwise_interaction": "interaction",
        "binary": "threshold",
        "logistic": "sigmoid",
        "sine": "oscillatory",
        "product": "tanh_product",
        "sparse": "sparse_nonlinear",
        "heavy_tail": "pareto_index",
        "heavy_tailed": "pareto_index",
        "pareto": "pareto_index",
    }

    def __init__(
        self,
        n,
        k,
        ky = 1,
        *,
        functional_form = "linear",
        function_params = None,
        column_covariance = None,
        predictor_distribution = "gaussian",
        predictor_df = 5.0,
        noise_scale = 0.0,
        rho = None,
        noise_type = "additive",
        center_latent = False,
        rng  = None,
        **kwargs,
    ) -> None:
        if n < 1:
            raise ValueError("n must be a positive integer.")
        if k < 1:
            raise ValueError("k must be a positive integer.")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative.")
        if noise_type not in {"additive", "multiplicative"}:
            raise ValueError("noise_type must be 'additive' or 'multiplicative'.")
        if noise_type == "additive" and rho is not None and not 0 < rho <= 1:
            raise ValueError("rho must satisfy 0 < rho <= 1 for additive noise.")
        if predictor_distribution not in {"gaussian", "student_t", "uniform_rdpg"}:
            raise ValueError(
                "predictor_distribution must be 'gaussian', 'student_t', or "
                "'uniform_rdpg'."
            )
        if predictor_distribution == "student_t" and predictor_df <= 0:
            raise ValueError("predictor_df must be positive for Student-t predictors.")

        self.n = int(n)
        self.kz = int(k)
        if ky != 1 and ky != self.kz:
            raise ValueError(
                "FunctionalGenerator supports either a scalar response (ky=1) "
                f"or a component-wise response (ky=k); received k={k}, ky={ky}."
            )
        self.ky = int(ky)
        self.functional_form = functional_form
        self.function_params = dict(function_params or {})
        self.predictor_distribution = predictor_distribution
        self.predictor_df = float(predictor_df)
        self.noise_scale = float(noise_scale)
        self.rho = rho
        self.noise_type = noise_type
        self.center_latent = bool(center_latent)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.extra_params = dict(kwargs)

        if self.predictor_distribution == "uniform_rdpg" and self.center_latent:
            raise ValueError(
                "center_latent must be False for uniform_rdpg because centering "
                "does not preserve valid RDPG inner products."
            ) 

        if column_covariance is None:
            column_covariance = np.eye(self.kz)
        self.column_covariance = np.asarray(column_covariance, dtype=float)
        self._validate_covariance()

        self._functional_name = self._normalize_functional_name(functional_form)
        if self.ky > 1 and self._functional_name in {"interaction", "hierarchical"}:
            raise ValueError(
                f"The '{self._functional_name}' functional cannot be applied "
                "component-wise to scalar inputs."
            )
        self.is_null = False

    def _normalize_functional_name(self, functional):
        if callable(functional):
            return None
        if not isinstance(functional, str):
            raise TypeError("functional must be a supported string or a callable.")
        name = functional.lower().strip()
        name = self._ALIASES.get(name, name)
        if name not in self.AVAILABLE_FUNCTIONALS:
            choices = ", ".join(self.AVAILABLE_FUNCTIONALS)
            raise ValueError(f"Unknown functional '{functional}'. Available: {choices}.")
        return name

    def _validate_covariance(self) -> None:
        if self.column_covariance.shape != (self.kz, self.kz):
            raise ValueError(
                "column_covariance must have shape "
                f"({self.kz}, {self.kz}), got {self.column_covariance.shape}."
            )
        if not np.allclose(self.column_covariance, self.column_covariance.T):
            raise ValueError("column_covariance must be symmetric.")
        eigenvalues = np.linalg.eigvalsh(self.column_covariance)
        if eigenvalues.min() < -1e-10:
            raise ValueError("column_covariance must be positive semidefinite.")

    def _sample_predictors(self) -> np.ndarray:
        """Draw a fresh ``(n, k)`` predictor matrix."""
        if self.predictor_distribution == "uniform_rdpg":
             # directions are sampled from a reflected gaussian and then normalsied
            # to get points we sample radius and multiply by direction  The radius and 
            # direction are sampled independently, which avoids rejection sampling in high k.
            direction = np.abs(self.rng.standard_normal((self.n, self.kz)))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            radius = self.rng.random(self.n) ** (1.0 / self.kz)
            return direction * radius[:, None]

        gaussian = self.rng.multivariate_normal(
            mean=np.zeros(self.kz),
            cov=self.column_covariance,
            size=self.n,
            check_valid="raise",
        )
        if self.predictor_distribution == "gaussian":
            Z = gaussian
        else:
            # The covariance matrix acts as a Student-t scale matrix here.
            chi2 = self.rng.chisquare(self.predictor_df, size=(self.n, 1))
            Z = gaussian * np.sqrt(self.predictor_df / chi2)

        if self.center_latent:
            Z = Z - Z.mean(axis=0, keepdims=True)
        return Z

    def _indey_weights(self) -> np.ndarray:
        """Get a normalized index using every coordinate by default."""
        raw_weights = self.function_params.get("weights")
        if raw_weights is None:
            weights = np.ones(self.kz, dtype=float) / np.sqrt(self.kz)
        else:
            weights = np.asarray(raw_weights, dtype=float)
            if weights.shape != (self.kz,):
                raise ValueError(
                    f"weights must have shape ({self.kz},), got {weights.shape}."
                )
            if not np.any(weights):
                raise ValueError("weights must not be identically zero.")

        if bool(self.function_params.get("normalize_index", True)):
            if self.predictor_distribution == "gaussian":
                variance = float(weights @ self.column_covariance @ weights)
            else:
                # For Student-t predictors, this normalization controls the
                # scale but does not turn the index into a standard normal.
                variance = float(weights @ self.column_covariance @ weights)
            if variance <= 0:
                raise ValueError("Index variance must be positive.")
            weights = weights / np.sqrt(variance)
        return weights

    def _index(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self._indey_weights()

    def _active_columns(self) -> np.ndarray:
        active = self.function_params.get("active_features")
        if active is None:
            count = int(self.function_params.get("active_count", min(5, self.kz)))
            if count < 1 or count > self.kz:
                raise ValueError("active_count must lie between 1 and kz.")
            return np.arange(count)

        active_indices = np.asarray(active, dtype=int)
        if active_indices.ndim != 1 or active_indices.size == 0:
            raise ValueError("active_features must be a non-empty 1D sequence.")
        if np.any(active_indices < 0) or np.any(active_indices >= self.kz):
            raise ValueError("active_features contains an out-of-range coordinate.")
        return active_indices

    def _evaluate_function(self, Z: np.ndarray) -> np.ndarray:
        """Evaluate the deterministic response ``Y=f(Z)``."""
        if callable(self.functional_form):
            y = self.functional_form(Z)
        else:
            method = getattr(self, f"_functional_{self._functional_name}")
            y = method(Z)

        y = np.asarray(y, dtype=float)
        if y.shape == (self.n, 1):
            y = y[:, 0]
        if y.shape != (self.n,):
            raise ValueError(
                "The functional must return shape (n,) or (n, 1); "
                f"received {y.shape}."
            )
        if not np.isfinite(y).all():
            raise FloatingPointError(
                "The functional generated non-finite response values. "
                "Adjust its parameters (for example, increase tail_clip)."
            )
        return y

    def _evaluate_componentwise(self, Z: np.ndarray) -> np.ndarray:
        """Reuse scalar functionals without changing the predictor generator.

        This temporary evaluation context uses the one-dimensional parameter
        contract and unit covariance.  No predictors are sampled through it,
        and the original generator's dimension and covariance stay unchanged.
        """
        scalar = copy(self)
        scalar.kz = scalar.ky = 1
        scalar.column_covariance = np.eye(1)
        return np.column_stack([
            scalar._evaluate_function(Z[:, [j]]) for j in range(self.kz)
        ])

    def _functional_linear(self, Z: np.ndarray) -> np.ndarray:
        return self._index(Z)

    def _functional_nonlinear_additive(self, Z: np.ndarray) -> np.ndarray:
        frequency = float(self.function_params.get("frequency", 1.0))
        quadratic_scale = float(self.function_params.get("quadratic_scale", 0.5))
        return (
            np.sin(frequency * Z) + quadratic_scale * Z**2
        ).sum(axis=1) / np.sqrt(self.kz)

    def _functional_interaction(self, Z: np.ndarray) -> np.ndarray:
        if self.kz < 2:
            raise ValueError("interaction requires kz >= 2.")
        pair_sum = 0.5 * (np.sum(Z, axis=1) ** 2 - np.sum(Z**2, axis=1))
        output_center = float(self.function_params.get("output_center", 0.0))
        default_sd = np.sqrt(self.kz * (self.kz - 1) / 2.0)
        output_sd = float(self.function_params.get("output_sd", default_sd))
        if not np.isfinite(output_center):
            raise ValueError("output_center must be finite for interaction.")
        if not np.isfinite(output_sd) or output_sd <= 0:
            raise ValueError("output_sd must be positive and finite for interaction.")
        return (pair_sum - output_center) / output_sd

    def _functional_radial(self, Z: np.ndarray) -> np.ndarray:
        return np.sum(Z**2, axis=1)

    def _functional_threshold(self, Z: np.ndarray) -> np.ndarray:
        threshold = float(self.function_params.get("threshold", 0.0))
        return (self._index(Z) > threshold).astype(float)

    def _functional_sigmoid(self, Z: np.ndarray) -> np.ndarray:
        threshold = float(self.function_params.get("threshold", 0.0))
        slope = float(self.function_params.get("slope", 1.0))
        global_scale = float(self.function_params.get("global_scale", 1))
        return expit(slope * (self._index(Z) - threshold)) * global_scale

    def _functional_piecewise(self, Z: np.ndarray) -> np.ndarray:
        switch_feature = int(self.function_params.get("switch_feature", 0))
        if not 0 <= switch_feature < self.kz:
            raise ValueError("switch_feature must be an integer in [0, k).")
        threshold = float(self.function_params.get("threshold", 0.0))
        base_index = self._index(Z)
        return np.where(Z[:, switch_feature] > threshold, base_index, -base_index)

    def _functional_oscillatory(self, Z: np.ndarray) -> np.ndarray:
        frequency = float(self.function_params.get("frequency", 5.0))
        global_scale = float(self.function_params.get("global_scale", 1.0))
        return np.sin(frequency * self._index(Z)) * global_scale

    def _functional_max(self, Z: np.ndarray) -> np.ndarray:
        return np.max(Z, axis=1)

    def _functional_abs_max(self, Z: np.ndarray) -> np.ndarray:
        center = np.asarray(self.function_params.get("center", 0.0), dtype=float)
        if center.ndim > 1 or (center.ndim == 1 and center.shape != (self.kz,)):
            raise ValueError(
                "center for abs_max must be a scalar or have shape "
                f"({self.kz},), got {center.shape}."
            )
        input_scale = float(self.function_params.get("input_scale", 1.0))
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("input_scale must be positive and finite for abs_max.")
        return np.max(np.abs(input_scale * (Z - center)), axis=1)

    def _functional_tanh_product(self, Z: np.ndarray) -> np.ndarray:
        scale = float(self.function_params.get("input_scale", 1.0))
        global_scale = float(self.function_params.get("global_scale", 1.0))
        center = np.asarray(self.function_params.get("center", 0.0), dtype=float)
        if center.ndim > 1 or (center.ndim == 1 and center.shape != (self.kz,)):
            raise ValueError(
                "center for tanh_product must be a scalar or have shape "
                f"({self.kz},), got {center.shape}."
            )
        return np.prod(np.tanh(scale * (Z - center)), axis=1) * global_scale

    def _functional_sparse_nonlinear(self, Z: np.ndarray) -> np.ndarray:
        active = self._active_columns()
        z_active = Z[:, active]
        frequency = float(self.function_params.get("frequency", 1.0))
        quadratic_scale = float(self.function_params.get("quadratic_scale", 1.0))
        return (
            np.sin(frequency * z_active) + quadratic_scale * z_active**2
        ).sum(axis=1) / np.sqrt(active.size)

    def _functional_hierarchical(self, Z: np.ndarray) -> np.ndarray:
        if self.kz < 4:
            raise ValueError("hierarchical requires kz >= 4.")
        y = Z[:, 0] * Z[:, 1] + 0.5 * Z[:, 2] ** 2 + np.sin(Z[:, 3])
        if self.kz > 4:
            linear_scale = float(self.function_params.get("linear_scale", 0.25))
            y = y + linear_scale * Z[:, 4:].sum(axis=1)
        return y

    def _functional_manifold(self, Z: np.ndarray) -> np.ndarray:
        w1 = self._indey_weights()
        raw_w2 = self.function_params.get("second_weights")
        if raw_w2 is None:
            w2 = (-1.0) ** np.arange(self.kz) / np.sqrt(self.kz)
        else:
            w2 = np.asarray(raw_w2, dtype=float)
            if w2.shape != (self.kz,):
                raise ValueError(
                    f"second_weights must have shape ({self.kz},), got {w2.shape}."
                )
        variance_w2 = float(w2 @ self.column_covariance @ w2)
        if variance_w2 <= 0:
            raise ValueError("The second manifold direction has zero variance.")
        w2 = w2 / np.sqrt(variance_w2)
        s1 = Z @ w1
        s2 = Z @ w2
        return s1**2 + np.sin(s2)

    def _functional_pareto_index(self, Z: np.ndarray) -> np.ndarray:
        """A deterministic Pareto response using every predictor by default.

        With Gaussian ``Z``, identity covariance, and ``center_latent=False``,
        the default normalized index is standard normal.  Therefore
        ``U = Phi(index)`` is Uniform(0, 1) and
        ``Y = scale * (1-U)^(-1/alpha)`` is Pareto(scale, alpha).
        """
        alpha = float(self.function_params.get("alpha", 1.5))
        scale = float(self.function_params.get("scale", 1.0))
        tail_clip = float(self.function_params.get("tail_clip", 1e-12))
        if alpha <= 0:
            raise ValueError("alpha must be positive for pareto_index.")
        if scale <= 0:
            raise ValueError("scale must be positive for pareto_index.")
        if not 0 < tail_clip < 0.5:
            raise ValueError("tail_clip must lie strictly between 0 and 0.5.")

        u = np.clip(ndtr(self._index(Z)), tail_clip, 1.0 - tail_clip)
        return scale * (1.0 - u) ** (-1.0 / alpha)

    def _add_noise(self, y: np.ndarray) -> np.ndarray:
        noise_scale = self.noise_scale
        if self.noise_type == "additive" and self.rho is not None:
            noise_var = np.var(y) * (1 - self.rho) / self.rho
            noise_scale = np.sqrt(noise_var)
        if noise_scale == 0.0:
            return y
        epsilon = self.rng.standard_normal(self.n)
        if self.noise_type == "additive":
            return y + noise_scale * epsilon
        return y * np.exp(noise_scale * epsilon)

    def sample_latent(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample predictors and their response.

        Returns
        -------
        Z : ndarray, shape (n, k)
            Vector-valued predictor samples.
        Y : ndarray, shape (n, ky)
            Scalar or component-wise response samples.
        """
        Z = self._sample_predictors()
        if self.ky == 1:
            y = self._add_noise(self._evaluate_function(Z))
            Y = y.reshape(self.n, 1)
        else:
            if not np.isfinite(Z).all():
                raise FloatingPointError("Predictor generation produced non-finite values.")
            signal = self._evaluate_componentwise(Z)
            Y = np.column_stack([
                self._add_noise(signal[:, j]) for j in range(self.ky)
            ])
        if not np.isfinite(Y).all():
            raise FloatingPointError("Noise generation produced non-finite values.")
        return Z, Y

    # Compatibility aliases for pipelines that use private sampler hooks.
    def _sample_latent_functional(self) -> tuple[np.ndarray, np.ndarray]:
        return self.sample_latent()

    def _sample_latent(self) -> tuple[np.ndarray, np.ndarray]:
        return self.sample_latent()

    def get_name(self) -> str:
        """Return a concise, filesystem-friendly description of this DGP."""
        name = self._functional_name if self._functional_name is not None else "custom"
        noise = (
            f"rho{self.rho:g}"
            if self.noise_type == "additive" and self.rho is not None
            else f"noise{self.noise_scale:g}"
        )
        response_dimension = f"ky{self.ky}_" if self.ky != 1 else ""
        return (
            f"functional_{name}_n{self.n}_k{self.kz}_"
            f"{response_dimension}"
            f"predictors_{self.predictor_distribution}_{noise}"
        )
