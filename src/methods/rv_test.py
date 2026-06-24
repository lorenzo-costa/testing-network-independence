
import numpy as np
from _base_class import BasePermutationTest, BaseEstimationMethod
from ..test_functions.rv_cca_coefficients import rv_coefficient, rv_coefficient_adjusted
from ..helper_functions.imhof import imhof

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class EstimateRV(BaseEstimationMethod):
    """Method to return RV coefficient between the latent positions of two networks"""
    def __init__(self, rng=None, solver=None, k=None, use_true_latent=False,
                 test_function=rv_coefficient, **kwargs):
        super().__init__(k=k, rng=rng, solver=solver, use_true_latent=use_true_latent)

        self.test_function = test_function

    def fit(self, data, **kwargs):
        """Estimate RV coefficient between the latent positions of two networks

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        self._process_input(data)
        self.test_stat_estimate = self.test_function(self.Zhat, self.Xhat)
        self.pvalue = None
        self.reject_null = None

    def get_name(self):
        return "EstimateRV"


class RVtest(BasePermutationTest):
    """Perform RV test for network independence.

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
        Type of permutation to use. Options:
        - latent: permute the estimated latent positions
        - observed: permute the observed data and re-estimate latent positions each time.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        approximation="permutation",
        k=None,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent_x=False,
        use_true_latent_z=False,    
        test_function=rv_coefficient_adjusted,
        permutation_type="latent",
        **kwargs,
    ):
        
        super().__init__(
            k=k,
            npermutations=npermutations,
            alpha=alpha,
            rng=rng,
            solver=solver,
            use_true_latent_x=use_true_latent_x,
            use_true_latent_z=use_true_latent_z,
            test_function=test_function,
            permutation_type=permutation_type,
        )

        self.approximation = approximation
        
    def fit(self, data, **kwargs):
        """Compute test statistic and p-value

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

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
        Xhat = self.Xhat.copy()
        Zhat = Zhat - Zhat.mean(axis=0)
        Xhat = Xhat - Xhat.mean(axis=0)
        n, _ = Zhat.shape

        rv = self.test_function(Zhat, Xhat)
        self.test_stat_estimate = rv

        # ── Plug-in eigenvalues of Ω̂_XZ ────────────────────────────────────────
        # Ω̂_XZ = (1/n) Σᵢ vec(X̃ᵢZ̃ᵢᵀ) vec(X̃ᵢZ̃ᵢᵀ)ᵀ  is (pq × pq)
        #
        # Its non-zero eigenvalues equal those of the (n × n) Gram matrix
        #   (1/n) G,  where  Gᵢⱼ = (X̃ᵢᵀX̃ⱼ)(Z̃ᵢᵀZ̃ⱼ) = (X̃X̃ᵀ)ᵢⱼ ⊙ (Z̃Z̃ᵀ)ᵢⱼ
        # so we never form the (pq × pq) object.

        GX = Xhat @ Xhat.T          # (n, n)  Gram matrix of X
        GZ = Zhat @ Zhat.T          # (n, n)  Gram matrix of Z
        Omega_gram = (GX * GZ) / n  # (n, n)  Hadamard product — Gram rep. of Ω̂

        hat_lambda = np.linalg.eigvalsh(Omega_gram)       # ascending
        hat_lambda = np.sort(hat_lambda)[::-1]            # descending
        # drop numerical zeros (rank ≤ min(n, p·q) in practice)
        hat_lambda = hat_lambda[hat_lambda > 1e-10 * hat_lambda[0]]

        # ── Normalisation: den = sqrt(tr(Ω̂²)) = sqrt(Σᵢ λ̂ᵢ²) ─────────────────
        den = np.sqrt(np.sum(hat_lambda ** 2))

        weights = hat_lambda / den
        self.pvalue = imhof(n * rv, weights)["Qq"]

    # def _fit_asymptotic(self):
    #     Zhat = self.Zhat.copy()
    #     Xhat = self.Xhat.copy()
    #     Zhat = Zhat - Zhat.mean(axis=0)
    #     Xhat = Xhat - Xhat.mean(axis=0)

    #     n, _ = Zhat.shape

    #     rv = self.test_function(Zhat, Xhat)

    #     SigmaXX = 1 / (n - 1) * Xhat.T @ Xhat
    #     SigmaZZ = 1 / (n - 1) * Zhat.T @ Zhat

    #     eigenvalues_X = np.linalg.eigvalsh(SigmaXX)
    #     eigenvalues_X = np.sort(eigenvalues_X)[::-1]  # sort in descending order
    #     eigenvalues_Z = np.linalg.eigvalsh(SigmaZZ)
    #     eigenvalues_Z = np.sort(eigenvalues_Z)[::-1]  # sort in descending order

    #     den = np.sqrt(np.trace(SigmaXX @ SigmaXX) * np.trace(SigmaZZ @ SigmaZZ))

    #     weights = (1+self.kappa) * np.outer(eigenvalues_X, eigenvalues_Z).flatten() / den 

    #     self.pvalue = imhof(n * rv, weights)["Qq"]

    def get_name(self):
        if self.approximation == "permutation":
            return "RV_PermutationTest_" + self.permutation_type
        else:
            return "RV_AsymptoticTest"