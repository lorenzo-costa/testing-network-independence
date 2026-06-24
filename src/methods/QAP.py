from ._base_class import BaseMethod
import numpy as np


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class QAP(BaseMethod):
    """Quadratic Assignment Procedure

    Parameters
    ----------
    alpha : float, optional
        The significance level (default 0.05)
    npermutations : int, optional
        The number of permutations for the test (default 100)
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        alpha=0.05,
        npermutations=100,
        null_hypothesis="independence",
        rng=None,
        **args,
    ):
        super().__init__()

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.alpha = alpha
        self.npermutations = npermutations
        self.null_hypothesis = null_hypothesis
        self.permutation_distribution = []
        
        # for consistency with other methods 
        self.X = None
        self.Z = None
        self.Xhat = None
        self.Zhat = None    

    def fit(self, data, **kwargs):
        """Estimates the latent positions and computes p-value

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B' i.e. adjacency matrices
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary with keys 'A', 'B'."
            )

        A = data.get("A")
        B = data.get("B")
        self.A = A
        self.B = B
        n = A.shape[0]

        self.test_stat_estimate = self._compute_test_stat(A, B)

        for i in range(self.npermutations):
            permutation = np.random.permutation(n)
            B_perm = B[permutation, :][:, permutation]
            test_stat_perm = self._compute_test_stat(A, B_perm)
            self.permutation_distribution.append(test_stat_perm)

        # compute pvalue
        self.pvalue = np.mean(
            np.abs(self.permutation_distribution) >= np.abs(self.test_stat_estimate)
        )

        self.reject_null = bool(self.pvalue < self.alpha)

        return

    def get_name(self):
        return "QAP"

    def _compute_test_stat(self, A, B):
        """Returns sqrt(n)rho if null hypothesis is independence (H0s) and sqrt(n)rho/v_w
        if null hypothesis is un-correlated (H0w)"""

        n = A.shape[0]
        A_centered = A - A.mean(axis=0)
        B_centered = B - B.mean(axis=0)
        A_centered[np.arange(n), np.arange(n)] = 0
        B_centered[np.arange(n), np.arange(n)] = 0

        phi_0_hat = 1 / (n * (n - 1) - 1) * np.sum(A_centered * B_centered)

        eta_hat_2_alpha = 1 / (n * (n - 1) - 1) * np.sum(A_centered**2)
        eta_hat_2_beta = 1 / (n * (n - 1) - 1) * np.sum(B_centered**2)

        rho_hat = phi_0_hat / np.sqrt(eta_hat_2_alpha * eta_hat_2_beta)

        if self.null_hypothesis == "independence":
            return np.sqrt(n) * rho_hat

        eta_hat_1_phi = (
            1 / n * np.sum((1 / (n - 1) * np.sum(A_centered * B_centered, axis=1)) ** 2)
        )

        v_w_hat = 4 * eta_hat_1_phi / (eta_hat_2_alpha * eta_hat_2_beta)

        return np.sqrt(n) * rho_hat / np.sqrt(v_w_hat)


