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
    
    def _sample_z(self):
        if self.z_distribution == 'uniform_rdpg':
            # Sample uniformly from the positive orthant of the unit Euclidean ball.
            # This is intended for BernoulliNetwork(rdpg=True): each sampled row has
            # norm at most one and nonnegative entries, so every pairwise inner
            # product is in [0, 1].
            z = self.rng.normal(size=(self.n, self.kz))
            z = np.abs(z)  # positive orthant
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            z /= np.maximum(norms, 1)  # project onto unit ball
        elif self.z_distribution == 'gaussian':
            z = self.rng.multivariate_normal(
                mean=np.zeros(self.kz), cov=self.column_covariance, size=self.n
            )
        else: 
            raise ValueError(f"Unsupported z_distribution: {self.z_distribution}")
        return z
    
    def _sample_x(self):
        if self.x_distribution == 'gaussian':
            x = self.rng.multivariate_normal(
                mean=np.zeros(self.kz), cov=self.column_covariance, size=self.n
            )
        elif self.x_distribution == 'cauchy':
            x = self.rng.standard_cauchy(size=(self.n, self.kz))
        elif self.x_distribution == 'uniform':
            x = self.rng.uniform(low=0, high=1, size=(self.n, self.kz))
        else:
            raise ValueError(f"Unsupported x_distribution: {self.x_distribution}")
        return x

    def sample_latent(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample predictors and their response.

        Returns
        -------
        Z : ndarray, shape (n, k)
            Vector-valued predictor samples.
        Y : ndarray, shape (n, ky)
            Scalar or component-wise response samples.
        """
        
        Z = self._sample_z()
        X = self._sample_x() 
        
        Y = (Z + X) % 1 + self.rng.normal(scale=self.noise_scale, size=(self.n, self.ky))
       
        return Z, Y, X

    def _sample_latent(self) -> tuple[np.ndarray, np.ndarray]:
        return self.sample_latent()
    
    def sample(self, x=None):
        if x is None:
            return self.sample_latent()
        else:
            Z = x
            Y = (Z + self.rng.uniform(size=Z.shape)) % 1 + self.rng.normal(scale=self.noise_scale, size=(self.n, self.ky))
            return Z, Y
    

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
