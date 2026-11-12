"""Canonical correlation test with permutation or Model-X CRT calibration."""

from ._base_class import BasePermutationTest
from ..test_functions.rv_cca_coefficients import first_cca_component


class CanonicalCorrelationTest(BasePermutationTest):
    """Test latent positions against observed Y using the first CCA component."""

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
        test_type="permutation",
        conditional_generator=None,
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
            test_type=test_type,
            conditional_generator=conditional_generator,
            **kwargs,
        )

    def fit(self, data, **kwargs):
        self._process_input(data)
        self._fit_null()

    def get_name(self):
        if self.test_type == "model_X":
            return "CCA_ModelXTest_covariate"
        return "CCA_PermutationTest_" + self.permutation_type
