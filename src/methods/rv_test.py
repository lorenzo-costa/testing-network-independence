import numpy as np
from ._base_class import BasePermutationTest, BaseEstimationMethod
from ..test_functions.rv_cca_coefficients import rv_coefficient, rv_coefficient_adjusted
from ..helper_functions.imhof import imhof

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class EstimateRV(BaseEstimationMethod):
    """Return the RV coefficient between network latent positions and observed Y."""

    def __init__(
        self,
        rng=None,
        solver=None,
        k=None,
        use_true_latent=False,
        test_function=rv_coefficient,
        **kwargs,
    ):
        super().__init__(k=k, rng=rng, solver=solver, use_true_latent=use_true_latent)

        self.test_function = test_function

    def fit(self, data, **kwargs):
        """Estimate the RV coefficient between tested Z and observed Y."""

        self._process_input(data)
        self.test_stat_estimate = self.test_function(self.Zhat, self.Y)
        self.pvalue = None
        self.reject_null = None

    def get_name(self):
        return "EstimateRV"


class RVTest(BasePermutationTest):
    """Test independence between a network latent representation and observed Y.

    Parameters
    ----------
    rho: float
        Correlation between latent positions (zero under independence i.e. H0 is true)
    npermutations : int
        Number of permutations to perform for the test.
    alpha : float
        Significance level for the test.
    solver : callable
        Function to estimate latent positions from the adjacency matrix.
    k : int
        Number of dimensions for the latent space.
    test_function : callable
        Function to compute the test statistic.
    permutation_type : str
        ``covariate`` permutes Y, ``latent`` permutes the tested Z, and
        ``observed`` relabels A and refits Z for each permutation.
    rng : np.random.Generator
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
        approximation="permutation",
        k=None,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent=False,
        test_function=rv_coefficient,
        permutation_type="covariate",
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
            test_function=test_function,
            permutation_type=permutation_type,
            one_sided=True,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

        self.approximation = approximation

    def fit(self, data, **kwargs):
        """Compute the test statistic and p-value."""

        self._process_input(data)

        if self.approximation == "permutation":
            self._fit_permutation()
        elif self.approximation == "asymptotic":
            self._fit_asymptotic()
        else:
            raise ValueError(
                "Invalid approximation method. Choose 'permutation' or 'asymptotic'."
            )

        self.reject_null = bool(self.pvalue < self.alpha)

        return

    def _fit_asymptotic(self):
        Zhat = self.Zhat.copy()
        Y = self.Y.copy()
        Zhat = Zhat - Zhat.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        n, _ = Zhat.shape

        rv = self.test_function(Zhat, Y)
        self.test_stat_estimate = rv

        GX = Y @ Y.T 
        GZ = Zhat @ Zhat.T  
        Omega_gram = (GX * GZ) / n  

        hat_lambda = np.linalg.eigvalsh(Omega_gram)  
        hat_lambda = np.sort(hat_lambda)[::-1]  
        hat_lambda = hat_lambda[hat_lambda > 1e-10 * hat_lambda[0]]

        den = np.sqrt(np.sum(hat_lambda**2))

        weights = hat_lambda / den
        self.pvalue = imhof(n * rv, weights)["Qq"]

    def get_name(self):
        if self.approximation == "permutation":
            return "RV_PermutationTest_" + self.permutation_type
        else:
            return "RV_AsymptoticTest"
