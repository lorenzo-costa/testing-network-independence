import numpy as np
from scipy.stats import norm

from ._base_class import BaseEstimationMethod, BasePermutationTest
from ..test_functions.ac_coefficient import ac_coefficient, asymptotic_unconditional_ac


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
            right_neighbor=self.use_right_neighbor,
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
            right_neighbor=self.use_right_neighbor,
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
        use_right_neighbor=False,
        block_size=2048,
        **kwargs,
    ):
        super().__init__(
            k=k, rng=rng, solver=solver, use_true_latent=use_true_latent
        )
        self.M = M
        self.use_permutation_coeff = use_permutation_coeff
        self.use_right_neighbor = use_right_neighbor
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
    """Permutation, Model-X, or asymptotic test using AC coefficients.

    With ``adaptive_m=True``, ``M`` is ignored. The test uses the rounded,
    capped, and deduplicated grid containing 1 and the powers of n with
    exponents 0.3, 0.5, 0.7, and 0.9. At each grid value it computes the
    coefficient for Y, which may be univariate or multivariate. All other
    coefficient and permutation options retain their usual behavior.
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
        use_right_neighbor=False,
        adaptive_m=False,
        adaptive_exponents=None,
        block_size=2048,
        n_jobs=1,
        batch_size=32,
        verbose=False,
        test_type="permutation",
        conditional_generator=None,
        _ignore_X=False, # temp for testing
        **kwargs,
    ):
        if "approximation" in kwargs:
            raise TypeError("'approximation' is no longer supported; use 'test_type'.")
        if test_type not in {"permutation", "model_X", "asymptotic"}:
            raise ValueError("test_type must be 'permutation', 'model_X', or 'asymptotic'.")
        if test_type == "model_X" and _ignore_X:
            raise ValueError("Model-X requires the AC statistic to use conditioning X.")
        if test_type == "asymptotic":
            if not use_permutation_coeff or use_right_neighbor or adaptive_m:
                raise ValueError(
                    "Asymptotic AC requires use_permutation_coeff=True, "
                    "use_right_neighbor=False, and adaptive_m=False."
                )
            if M is not None and (
                isinstance(M, (bool, np.bool_))
                or not isinstance(M, (int, np.integer))
            ):
                raise ValueError("Asymptotic AC requires a fixed integer M.")
        self.M = M
        self.use_permutation_coeff = use_permutation_coeff
        self.use_right_neighbor = use_right_neighbor
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
            # The shared resampling base has no asymptotic fitting path.
            test_type="permutation" if test_type == "asymptotic" else test_type,
            conditional_generator=conditional_generator,
        )
        self.test_type = test_type

    def fit(self, data, **kwargs):
        self._process_input(data)
        if self.test_type == "asymptotic":
            self._fit_asymptotic()
            return
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

    def _fit_asymptotic(self):
        if self.X is not None and not self._ignore_X:
            raise ValueError("Asymptotic AC requires unconditional X=None.")
        coefficient, variance = asymptotic_unconditional_ac(
            self.Y, self.Zhat, M=1 if self.M is None else self.M,
            rng=self.rng, block_size=self.block_size,
        )
        self.test_stat_estimate = coefficient
        self.asymptotic_variance = variance
        self.asymptotic_z_statistic = np.sqrt(len(self.Y)) * coefficient / np.sqrt(variance)
        self.pvalue = float(norm.sf(self.asymptotic_z_statistic))
        self.reject_null = bool(self.pvalue < self.alpha)

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
        if self.adaptive_m:
            return self._adaptive_statistics(Z, Y, rng=rng)
        return self._ac_statistic(Z, Y, rng=rng)

    def get_name(self):
        if self.test_type == "asymptotic":
            return f"MultivariateAC_AsymptoticTest_{1 if self.M is None else self.M}"
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
        return name
