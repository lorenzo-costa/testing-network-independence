import numpy as np
from ._base_class import BasePermutationTest, BaseEstimationMethod
from ..test_functions.ac_coefficient import ac_coefficient

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class EstimateAC(BaseEstimationMethod):
    """Method to return AC coefficient between the latent positions of two networks"""

    def __init__(
        self,
        rng=None,
        solver=None,
        k=None,
        use_true_latent=False,
        M=1,
        test_function=ac_coefficient,
        **kwargs,
    ):
        super().__init__(k=k, rng=rng, solver=solver, use_true_latent=use_true_latent)
        self.M = M
        self.test_function = test_function

    def fit(self, data, **kwargs):
        """Estimate AC coefficient between the latent positions of two networks

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        self._process_input(data)

        # X takes the role of response variable Y in the AC function, Z is the predictor.
        # for consistency keep the names as X and Z.
        # in _process_input when use_true_latent is True, Zhat copies true Z
        self.test_stat_estimate = self.test_function(
            Y=self.X, Z=self.Zhat, M=self.M, rng=self.rng
        )
        self.pvalue = None
        self.reject_null = None

    def get_name(self):
        return "EstimateAC"


class MultivariateACTest(BasePermutationTest):
    """Testing independence using the multivariate AC coefficient

     Reference:

     Parameters
     ----------
     npermutations : int
        Number of permutations for significance testing.
    alpha : float
        Significance level for hypothesis testing.
    solver : callable
        Function to estimate latent positions from the adjacency matrix.
    k : int
        Number of dimensions for the latent space.
    use_true_latent : bool
        Whether to use the true latent positions (if True, Z and X must be provided in data)
    M : int
        Number of nearest neighbors to use.
    """

    def __init__(
        self,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent_x=False,
        use_true_latent_z=False,
        right_neighbor=False,
        M=1,
        k=None,
        **kwargs,
    ):
        super().__init__(
            k=k,
            npermutations=npermutations,
            alpha=alpha,
            use_true_latent_x=use_true_latent_x,
            use_true_latent_z=use_true_latent_z,
            solver=solver,
            test_function=ac_coefficient,
            rng=rng,
        )

        self.M = M
        self.right_neighbor = right_neighbor

    def fit(self, data, **kwargs):
        """Compute multivariate AC coefficient. X is used as response variables (Y),
        Z is the predictor. If use_true_latent is False, the function estimates
        the latent position Zhat from adj matrix A.

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        use_true_latent : bool
            Whether to use the true latent positions (if True, Z and X must be provided in data)
             or to estimate them from the adjacency matrix (if False, A must be provided in data).
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary with keys 'A', 'B'."
            )
        if "X" not in data.keys():
            raise ValueError(
                "True positions X must be provided for multivariate AC test."
            )

        self._process_input(data)

        # X takes the role of response variable Y in the AC function, Z is the predictor.
        # for consistency keep the names as X and Z.
        # in _process_input when use_true_latent is True, Zhat and Xhat copy true latent
        self.test_stat_estimate = self.test_function(
            Y=self.Xhat,
            Z=self.Zhat,
            M=self.M,
            rng=self.rng,
            right_neighbor=self.right_neighbor,
        )

        for _ in range(self.npermutations):
            perm = self.rng.permutation(self.Zhat.shape[0])
            Zhat_perm = self.Zhat[perm, :]
            test_stat_perm = self.test_function(
                Y=self.Xhat,
                Z=Zhat_perm,
                M=self.M,
                rng=self.rng,
                right_neighbor=self.right_neighbor,
            )
            self.permutation_distribution.append(test_stat_perm)

        # get and store pvalue
        self.pvalue = np.mean(
            np.abs(self.permutation_distribution) >= np.abs(self.test_stat_estimate)
        )

        self.reject_null = bool(self.pvalue < self.alpha)

    def get_name(self):
        return "MultivariateAC_PermutationTest_" + str(self.M)
