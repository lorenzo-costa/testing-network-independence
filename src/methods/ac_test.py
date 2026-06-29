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
        test_function=ac_coefficient,
        
        M=None, 
        aggregate_coeff=None, 
        use_permutation_coeff=False, 
        use_right_neighbor=False, 
        block_size=2048,
        **kwargs,
    ):
        super().__init__(k=k, rng=rng, solver=solver, use_true_latent=use_true_latent)
        self.M = M
        self.test_function = test_function
        self.aggregate_coeff = aggregate_coeff
        self.use_permutation_coeff = use_permutation_coeff
        self.use_right_neighbor = use_right_neighbor
        self.block_size = block_size

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
            Y=self.X, 
            Z=self.Zhat, 
            M=self.M, 
            rng=self.rng, 
            aggregate=self.aggregate_coeff,
            permutation=self.use_permutation_coeff,
            right_neighbor=self.use_right_neighbor,
            block_size=self.block_size
        )
        self.pvalue = None
        self.reject_null = None

    def get_name(self):
        return "EstimateAC"


class MultivariateACTest(BasePermutationTest):
    """Testing independence using the multivariate AC coefficient

     Parameters
     ----------
    k : int
        Embedding dimension to use (not necessarily the true one but one used for the test)
    M : int
        Number of nearest neighbours to use for test
    use_true_latent_x : bool
        Whether to use the true latent positions for X (if True, X must be provided in data)
    use_true_latent_z : bool
        Whether to use the true latent positions for Z (if True, Z must be provided in data)
    aggregate_coeff : bool
        If None, use AC coeff with a single value of M. If 'avg' or 'max' aggregate AC coeff
        across all valued of M in the interval [1, M] using the specified aggregation method.
    use_right_neighbor : bool
        If True, use the right neighbor of each point to compute the AC coefficient. 
        Available only if both are univariate
    use_permutation_coeff : bool
        If True, use the permutation version of the AC coefficient. Used only for 
        multivariate response. 
    npermutations : int
        Number of permutations to use for the permutation test
    alpha : float
        Significance level for the test
    rng : np.random.Generator
        Random number generator for reproducibility
    """

    def __init__(
        self,
        k=None,
        M=None, 
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent_x=False,
        use_true_latent_z=False,
        aggregate_coeff=None, 
        use_permutation_coeff=False, 
        use_right_neighbor=False, 
        block_size=2048,
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
        self.aggregate_coeff = aggregate_coeff
        self.use_permutation_coeff = use_permutation_coeff
        self.use_right_neighbor = use_right_neighbor
        self.block_size = block_size

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
            aggregate=self.aggregate_coeff,
            permutation=self.use_permutation_coeff,
            right_neighbor=self.use_right_neighbor,
            block_size=self.block_size
        )

        for _ in range(self.npermutations):
            perm = self.rng.permutation(self.Zhat.shape[0])
            Zhat_perm = self.Zhat[perm, :]
            test_stat_perm = self.test_function(
                Y=self.Xhat,
                Z=Zhat_perm,
                M=self.M,
                rng=self.rng,
                aggregate=self.aggregate_coeff,
                permutation=self.use_permutation_coeff,
                right_neighbor=self.use_right_neighbor,
                block_size=self.block_size
            )
            self.permutation_distribution.append(test_stat_perm)

        # get and store pvalue
        self.pvalue = np.mean(
            np.abs(self.permutation_distribution) >= np.abs(self.test_stat_estimate)
        )

        self.reject_null = bool(self.pvalue < self.alpha)

    def get_name(self):
        return "MultivariateAC_PermutationTest_" + str(self.M) + "_" + str(self.aggregate_coeff)
