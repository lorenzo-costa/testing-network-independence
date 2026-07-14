from ._base_class import BaseEstimationMethod, BasePermutationTest
from ..test_functions.ac_coefficient import ac_coefficient


class _ACMixin:
    def _ac_statistic(self, Z, Y):
        return ac_coefficient(
            Y=Y,
            Z=Z,
            X=self.X if not self._ignore_X else None,
            M=self.M,
            rng=self.rng,
            aggregate=self.aggregate_coeff,
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
        aggregate_coeff=None,
        use_permutation_coeff=False,
        use_right_neighbor=False,
        block_size=2048,
        **kwargs,
    ):
        super().__init__(
            k=k, rng=rng, solver=solver, use_true_latent=use_true_latent
        )
        self.M = M
        self.aggregate_coeff = aggregate_coeff
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
    """Permutation test using the multivariate AC coefficient."""

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
        aggregate_coeff=None,
        use_permutation_coeff=False,
        use_right_neighbor=False,
        block_size=2048,
        _ignore_X=False, # temp for testing
        **kwargs,
    ):
        self.M = M
        self.aggregate_coeff = aggregate_coeff
        self.use_permutation_coeff = use_permutation_coeff
        self.use_right_neighbor = use_right_neighbor
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
            stratify_permutations=not _ignore_X,
        )

    def fit(self, data, **kwargs):
        self._process_input(data)
        self._fit_permutation()

    def get_name(self):
        if self._ignore_X:
            return (
                "MultivariateAC_PermutationTest_"
                + self.permutation_type
                + "_"
                + str(self.M)
                + "_"
                + str(self.aggregate_coeff)
                + "_ignoreX"
            )
        else:
            return (
                "MultivariateAC_PermutationTest_"
                + self.permutation_type
                + "_"
                + str(self.M)
                + "_"
                + str(self.aggregate_coeff)
            )
