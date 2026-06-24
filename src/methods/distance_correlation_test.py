import numpy as np
from ._base_class import BasePermutationTest, BaseEstimationMethod
from scipy.spatial.distance import pdist, squareform
import warnings
from scipy.stats import multiscale_graphcorr


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class DistanceCorrelationTest(BasePermutationTest):
    """Implementation of Diffusion Correlation algorithm.

    Parameters
    ----------
    k : int
        Dimensionality of the latent space.
    test_method : str
        Statistical test method to use. Options: "mgc", "dcorr".
    npermutations : int
        Number of permutations for significance testing.
    alpha : float
        Significance level for hypothesis testing.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        k=None,
        test_method="mgc",
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent_x=False,
        use_true_latent_z=False,
        **kwargs,
    ):
        super().__init__(
            k=k,
            npermutations=npermutations,
            alpha=alpha,
            rng=rng,
            solver=solver,
            test_function=lambda x: x,
            use_true_latent_x=use_true_latent_x,
            use_true_latent_z=use_true_latent_z,
        )

        self.rng = np.random.default_rng() if rng is None else rng

        self.test_method = test_method

        self.eps = 1e-10

    def compute_distance_matrix(self, U):
        return squareform(pdist(U, metric="euclidean"))

    def _double_center(self, D):
        """
        Double centering of distance matrix
        """
        n = D.shape[0]
        row_mean = np.mean(D, axis=1, keepdims=True)
        col_mean = np.mean(D, axis=0, keepdims=True)
        total_mean = np.mean(D)

        return D - row_mean - col_mean + total_mean

    def fit(self, data):
        """Compute p-value using permutation test

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        self._process_input(data)

        distances_A = self.compute_distance_matrix(self.Xhat)
        distances_B = self.compute_distance_matrix(self.Zhat)

        if self.test_method == "mgc":
            # get rid of warning for number of permutations too small
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    out_mgc = multiscale_graphcorr(
                        distances_A,
                        distances_B,
                        random_state=self.rng,
                        reps=self.npermutations,
                    )
                    pvalue = out_mgc.pvalue
                    test_stat_estimate = out_mgc.statistic

                except IndexError:
                    pvalue = 1.0
                    print("Error in computing MGC. Check the distance matrices.")
                    print(distances_A.sum(), distances_B.sum())
        else:
            print(self.test_method)
            raise ValueError("Unknown method for computing test statistic.")

        self.pvalue = pvalue
        self.test_stat_estimate = test_stat_estimate

        self.reject_null = bool(self.pvalue < self.alpha)

    def get_name(self):
        return "DistanceCorrelation"
