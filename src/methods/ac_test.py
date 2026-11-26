import numpy as np
from scipy.stats import norm

from ._base_class import BaseEstimationMethod, BasePermutationTest
from ..test_functions.ac_coefficient import ac_coefficient, asymptotic_unconditional_ac
from ..test_functions._ac_helpers import _validate_m


class _ACMixin:
    def _conditioning_x(self):
        if getattr(self, "_ignore_X", False):
            return None
        if getattr(self, "estimate_conditional", False):
            return self.X[self.test_indices]
        return self.X

    def _ac_statistic(self, Z, Y, rng=None):
        rng = self.rng if rng is None else rng
        return ac_coefficient(
            Y=Y,
            Z=Z,
            X=self._conditioning_x(),
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
            X=self._conditioning_x(),
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

    def fit(self, data, **kwargs):
        self._process_input(data)
        self.test_stat_estimate = self._ac_statistic(self.Zhat, self.Y)
        self.pvalue = None
        self.reject_null = None

    def get_name(self):
        return "EstimateAC"


class MultivariateACTest(_ACMixin, BasePermutationTest):
    """Permutation, Model-X, or asymptotic test using AC coefficients.

    A float ``M`` below one selects ``round(n**M)`` neighbors for either
    test type. The asymptotic calibration is derived for fixed neighbor counts.
    Scalar Y uses the rank coefficient and does not need a shuffled reference;
    multivariate Y requires ``use_permutation_coeff=True``.

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
        standardize=False,
        _ignore_X=False,
        **kwargs,
    ):
        if test_type not in {"permutation", "model_X", "asymptotic"}:
            raise ValueError("test_type must be 'permutation', 'model_X', or 'asymptotic'.")
        if test_type == "model_X" and _ignore_X:
            raise ValueError("Model-X requires the AC statistic to use conditioning X.")
        if test_type == "asymptotic":
            if adaptive_m:
                raise ValueError("Asymptotic AC requires adaptive_m=False.")
            if M is not None and (
                isinstance(M, (bool, np.bool_))
                or not isinstance(M, (int, np.integer, float, np.floating))
                or (
                    isinstance(M, (float, np.floating))
                    and (not np.isfinite(M) or M >= 1)
                )
            ):
                raise ValueError(
                    "Asymptotic AC requires an integer M or a finite float "
                    "smaller than 1."
                )
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
        self._ignore_X = bool(_ignore_X)
        super().__init__(
            k=k,
            npermutations=npermutations,
            alpha=alpha,
            use_true_latent=use_true_latent,
            permutation_type=permutation_type,
            solver=solver,
            test_function=self._ac_statistic,
            rng=rng,
            stratify_permutations=test_type == "permutation" and not self._ignore_X,
            one_sided=True,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            # The shared resampling base has no asymptotic fitting path.
            test_type="permutation" if test_type == "asymptotic" else test_type,
            conditional_generator=conditional_generator,
            **kwargs,
        )
        self.test_type = test_type

    def fit(self, data, **kwargs):
        self._process_input(data)
        if self.estimate_conditional:
            self._validate_method_input()
        tested_z = self.Zhat[self.test_indices] if self.estimate_conditional else self.Zhat
        self.standardization_scale_ = 1.0
        if self.standardize:
            with np.errstate(over="ignore", invalid="ignore"):
                scale = float(np.sqrt(np.mean(np.var(tested_z, axis=0))))
            if not np.isfinite(scale):
                raise ValueError("Z block standard deviation must be finite to standardize.")
            if scale > 0:
                self.standardization_scale_ = scale
        if self.test_type == "asymptotic":
            self._fit_asymptotic()
            return
        if self.adaptive_m:
            if self.npermutations < 1:
                raise ValueError(
                    "Adaptive M selection requires at least one null draw to "
                    "compute sample standard deviations."
                )
            self.adaptive_m_values = self._adaptive_m_grid(tested_z.shape[0])
        self._fit_null()

        if self.adaptive_m: 
            self.adaptive_s_statistics = self.statistic_matrix
            self.adaptive_m_means = self.statistic_means
            self.adaptive_m_stds = self.statistic_standard_deviations
            self.adaptive_z_statistics = self.standardized_statistics

    def _fit_asymptotic(self):
        if self.X is not None and not self._ignore_X:
            raise ValueError("Asymptotic AC requires unconditional X=None.")
        if self.Y.shape[1] > 1 and not self.use_permutation_coeff:
            raise ValueError(
                "Asymptotic AC with multivariate Y requires "
                "use_permutation_coeff=True."
            )
        self.effective_M = _validate_m(
            1 if self.M is None else self.M, len(self.Y)
        )
        tested_z = self.Zhat / self.standardization_scale_ if self.standardize else self.Zhat
        coefficient, variance = asymptotic_unconditional_ac(
            self.Y, tested_z, M=self.effective_M,
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
        if self.standardize:
            Z = Z / self.standardization_scale_
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
        if self.standardize:
            name += "_standardized"
        if self._ignore_X:
            name += "_ignoreX"
        return name
