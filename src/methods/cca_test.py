import numpy as np
from ._base_class import BasePermutationTest, BaseEstimationMethod
from ..test_functions.rv_cca_coefficients import first_cca_component


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# TODO: add estimation method


class CanonicalCorrelationTest(BasePermutationTest):
    """Testing independence using Canonical Correlation Analysis

    Reference: Fuchs and Keith Levin 2025

    Parameters
    ----------
    rho : float
        Correlation between latent positions (zero under independence i.e. H0 is true)
    k : int
        Dimensionality of the latent space.
    npermutations : int
        Number of permutations for significance testing.
    alpha : float
        Significance level for hypothesis testing.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    n_jobs : int
        Number of permutation worker processes. ``-1`` uses all available CPUs.
    batch_size : int
        Target number of work batches per permutation worker.
    verbose : bool
        Whether to display permutation progress.
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
            permutation_type=permutation_type,
            use_true_latent=use_true_latent,
            solver=solver,
            test_function=first_cca_component,
            rng=rng,
            one_sided=True,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def fit(self, data, **kwargs):
        """Compute test statistic and p-value

        Parameters
        ----------
        data : dict
            A dictionary containing observed ``Y`` and either true ``Z`` or
            adjacency matrix ``A``.
        """

        self._process_input(data)

        self._fit_permutation()

        self.reject_null = bool(self.pvalue < self.alpha)

        return

    def get_name(self):
        return "CCA_PermutationTest_" + self.permutation_type
