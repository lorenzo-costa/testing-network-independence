import numpy as np

from ._base_class import BaseEstimationMethod, BasePermutationTest
from ..test_functions.ac_coefficient import ac_coefficient


class _ACMixin:
    def _ac_statistic(self, Z, Y, rng=None):
        rng = self.rng if rng is None else rng
        return ac_coefficient(
            Y=Y,
            Z=Z,
            X=self.X if not self._ignore_X else None,
            M=self.M,
            rng=rng,
            permutation=self.use_permutation_coeff,
            block_size=self.block_size,
        )

    def _adaptive_s_statistic(self, Z, Y, M, rng=None):
        """Return the coefficient xi_M(Y, Z) for one M."""
        rng = self.rng if rng is None else rng
        return ac_coefficient(
            Y=Y,
            Z=Z,
            X=self.X if not self._ignore_X else None,
            M=M,
            rng=rng,
            permutation=self.use_permutation_coeff,
            block_size=self.block_size,
        )


class EstimateAC(_ACMixin, BaseEstimationMethod):
    """Estimate the AC coefficient between network latent Z and observed Y."""

    def __init__(
        self,
        rng=None,
        solver=None,
        k=None,
        use_true_latent=False,
        M=None,
        use_permutation_coeff=False,
        block_size=2048,
        **kwargs,
    ):
        super().__init__(
            k=k, rng=rng, solver=solver, use_true_latent=use_true_latent
        )
        self.M = M
        self.use_permutation_coeff = use_permutation_coeff
        self.block_size = block_size
        self._ignore_X = False

    def fit(self, data, **kwargs):
        self._process_input(data)
        self.test_stat_estimate = self._ac_statistic(self.Zhat, self.Y)
        self.pvalue = None
        self.reject_null = None

    def get_name(self):
        return "EstimateAC"


class MultivariateACTest(_ACMixin, BasePermutationTest):
    """Permutation or Model-X test using fixed-M or adaptive AC coefficients.

    With ``adaptive_m=True``, ``M`` is ignored. The test uses the rounded,
    capped, and deduplicated grid containing 1 and the powers of n with
    exponents 0.3, 0.5, 0.7, and 0.9. At each grid value it computes the
    coefficient for Y, which may be univariate or multivariate. All other
    coefficient and permutation options retain their usual behavior.

    With ``standardize=True``, divide the tested Z block by
    ``sqrt(mean(var(Z, axis=0, ddof=0)))`` before finding neighbors. X and Y
    are unchanged, and Z is not centered. This single block scale preserves
    rotations of the embedding. The fitted ``standardization_scale_`` is
    held fixed for the observed statistic and all null draws, including
    every adaptive-M component. A constant block uses scale 1.
    """

    def __init__(
        self,
        k=None,
        M=None,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent=False,
        permutation_type="covariate",
        use_permutation_coeff=False,
        adaptive_m=False,
        adaptive_exponents=None,
        block_size=2048,
        n_jobs=1,
        batch_size=32,
        verbose=False,
        test_type="permutation",
        conditional_generator=None,
        _ignore_X=False, # temp for testing
        standardize=False,
        **kwargs,
    ):
        if "approximation" in kwargs:
            raise TypeError("'approximation' is no longer supported; use 'test_type'.")
        if test_type == "model_X" and _ignore_X:
            raise ValueError("Model-X requires the AC statistic to use conditioning X.")
        if not isinstance(standardize, (bool, np.bool_)):
            raise ValueError("standardize must be a boolean.")
        self.standardize = bool(standardize)
        self.standardization_scale_ = 1.0
        self.M = M
        self.use_permutation_coeff = use_permutation_coeff
        self.adaptive_m = bool(adaptive_m)
        self.adaptive_exponents = (adaptive_exponents
            if adaptive_exponents is not None
            else (0.3, 0.5, 0.7, 0.9))
        
        self.block_size = block_size
        self._ignore_X = _ignore_X
        super().__init__(
            k=k,
            npermutations=npermutations,
            alpha=alpha,
            use_true_latent=use_true_latent,
            permutation_type=permutation_type,
            solver=solver,
            test_function=self._ac_statistic,
            rng=rng,
            stratify_permutations=test_type == "permutation" and not _ignore_X,
            one_sided=True,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            test_type=test_type,
            conditional_generator=conditional_generator,
        )

    def fit(self, data, **kwargs):
        self._process_input(data)
        self.standardization_scale_ = 1.0
        if self.standardize:
            with np.errstate(over="ignore", invalid="ignore"):
                scale = float(np.sqrt(np.mean(np.var(self.Zhat, axis=0))))
            if not np.isfinite(scale):
                raise ValueError("Z block standard deviation must be finite to standardize.")
            if scale > 0:
                self.standardization_scale_ = scale
        if self.adaptive_m:
            if self.npermutations < 1:
                raise ValueError(
                    "Adaptive M selection requires at least one null draw to "
                    "compute sample standard deviations."
                )
            self.adaptive_m_values = self._adaptive_m_grid(self.Y.shape[0])
        self._fit_null()

        if self.adaptive_m: 
            self.adaptive_s_statistics = self.statistic_matrix
            self.adaptive_m_means = self.statistic_means
            self.adaptive_m_stds = self.statistic_standard_deviations
            self.adaptive_z_statistics = self.standardized_statistics

    def _adaptive_m_grid(self, n):
        """Return the adaptive M grid, always including one nearest neighbor."""
        capped = (
            min(round(n**exponent), n - 1)
            for exponent in self.adaptive_exponents
        )
        return np.asarray(list(dict.fromkeys([1, *capped])), dtype=int)

    def _adaptive_statistics(self, Z, Y, rng=None):
        return np.asarray(
            [
                self._adaptive_s_statistic(Z, Y, M, rng=rng)
                for M in self.adaptive_m_values
            ],
            dtype=float,
        )

    def _evaluate_test_statistic(self, Z, Y, rng):
        if self.standardize:
            Z = Z / self.standardization_scale_
        if self.adaptive_m:
            return self._adaptive_statistics(Z, Y, rng=rng)
        return self._ac_statistic(Z, Y, rng=rng)

    def get_name(self):
        if self.adaptive_m:
            statistic_description = "adaptive"
        else:
            statistic_description = str(self.M)

        test_label = "PermutationTest" if self.test_type == "permutation" else "ModelXTest"
        name = (
            f"MultivariateAC_{test_label}_"
            f"{self.permutation_type}_{statistic_description}"
        )
        if self._ignore_X:
            name += "_ignoreX"
        if self.standardize:
            name += "_standardized"
        return name
