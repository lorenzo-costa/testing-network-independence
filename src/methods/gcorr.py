"""Test based on graph correlation from Wen et al. (2023) RKHS-based Latent 
Position Random Graph Correlation"""

import numpy as np
from ._base_class import BasePermutationTest
from ..test_functions.graph_correlation import gcor

class GraphCorrelation(BasePermutationTest):
    """Testing independence using Graph Correlation
    
    Reference: Wen et al. (2023) RKHS-based Latent Position Random Graph Correlation

    Parameters
    ----------
    rho : float
        Correlation between latent positions (zero under independence i.e. H0 is true)
    d_y, d_x : int
        Embedding dimensions of Y and each X network.
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
        permutation_type="latent",
        d_y=None,
        d_x=None,
        n_jobs=1,
        batch_size=32,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            d_y=d_y,
            d_x=d_x,
            npermutations=npermutations,
            alpha=alpha,
            permutation_type=permutation_type,
            use_true_latent=use_true_latent,
            solver=solver,
            test_function=gcor,
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
            A dictionary containing ``A_Y`` and a list ``A_X`` of networks,
            optionally including true latent ``Y`` and a list ``X``.
        """

        self._process_input(data)

        self._fit_permutation()

        self.reject_null = bool(self.pvalue < self.alpha)

        return

    def _evaluate_test_statistic(self, Y, X, rng):
        """Evaluate Graph Correlation while reusing the fixed X-side decomposition."""
        return gcor(Y, X)

    def get_name(self):
        return "GraphCorrelation_PermutationTest_" + self.permutation_type