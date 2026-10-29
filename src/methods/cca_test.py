"""Permutation test based on the first canonical correlation."""

from ._base_class import BasePermutationTest
from ..test_functions.rv_cca_coefficients import first_cca_component


class CanonicalCorrelationTest(BasePermutationTest):
    """Test independence of network latent positions and observed covariates.

    The statistic is the largest unregularized canonical correlation. The
    shared permutation engine supplies the one-sided Monte Carlo p-value.
    """

    def __init__(
        self,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent=False,
        permutation_type="covariate",
        k=None,
        n_jobs=1,
        batch_size=32,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            k=k,
            npermutations=npermutations,
            alpha=alpha,
            rng=rng,
            solver=solver,
            use_true_latent=use_true_latent,
            test_function=first_cca_component,
            permutation_type=permutation_type,
            one_sided=True,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )

    def fit(self, data, **kwargs):
        """Compute the observed coefficient and permutation p-value."""
        self._process_input(data)
        self._fit_permutation()

    def get_name(self):
        return "CCA_PermutationTest_" + self.permutation_type
