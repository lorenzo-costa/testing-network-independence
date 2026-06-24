import numpy as np
from ._base_class import BaseMethod


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class ObservedCVM(BaseMethod):
    def __init__(
        self,
        npermutations=100,
        alpha=0.05,
        rng=None,
        use_true_latent=False,
        test_function=None,
        permutation_type="latent",
        **kwargs,
    ):
        super().__init__()

        self.permutation_distribution = []

        self.alpha = alpha
        self.npermutations = npermutations
        self.rng = np.random.default_rng() if rng is None else rng

        self.test_function = test_function
        self.permutation_type = permutation_type
        self.use_true_latent = use_true_latent

    def fit(self, data, **kwargs):
        """Get null distribution of RV coefficient with permutations

        The function estimates the latent position of the networks independently,
        computes the RV coefficient and obtains the p-value by permutation.

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary with keys 'A', 'B'."
            )

        # need to estimate latent positions
        A = data.get("A")
        B = data.get("B")
        Z = data.get("Z", None)
        X = data.get("X", None)

        # for consistency with other methods
        self.Z = Z
        self.X = X
        self.Zhat = None
        self.Xhat = None

        self.A = A
        self.B = B

        matrix_A = Z @ Z.T if self.use_true_latent and Z is not None else A
        matrix_B = X @ X.T if self.use_true_latent and X is not None else B

        test_stat_estimate = self.test_function(matrix_A, matrix_B)
        self.test_stat_estimate = test_stat_estimate
        for _ in range(self.npermutations):
            perm = self.rng.permutation(B.shape[0])
            matrix_B_perm = matrix_B[perm][:, perm]
            test_stat_perm = self.test_function(matrix_A, matrix_B_perm)
            self.permutation_distribution.append(test_stat_perm)

        # compute pvalue
        pvalue = np.mean(
            [i >= self.test_stat_estimate for i in self.permutation_distribution]
        )
        self.pvalue = pvalue
        self.reject_null = bool(self.pvalue < self.alpha)

        return

    def get_name(self):
        return "ObservedCVMPermutationTest"
