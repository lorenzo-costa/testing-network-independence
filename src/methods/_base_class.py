import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class BaseMethod:
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_estimated(self):
        """Get fit results

        Returns
        -------
        A dictionary with 'estimated_latent', 'true_latent', 'p-value', 'reject_null', and 'null' keys.
        """
        results = {
            "estimated_latent": (self.Xhat, self.Zhat),
            "true_latent": (self.X, self.Z),
            "p-value": self.pvalue,
            "reject_null": self.reject_null,
            "test_stat": self.test_stat_estimate,
        }
        return results


class BaseEstimationMethod(BaseMethod):
    """Base class for estimation methods"""

    def __init__(self, rng=None, solver=None, k=None, use_true_latent=False, **kwargs):
        super().__init__()

        self.rng = rng if rng is not None else np.random.default_rng()
        if solver is None:
            raise ValueError("Solver must be provided")
        self.solver = solver
        self.k = k
        self.use_true_latent = use_true_latent

    def _process_input(self, data):
        """Utility function to extract and process input data, estimate latent
        positions if needed, and store results in the object."""

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary with keys 'A', 'B'."
            )
        if self.use_true_latent:
            if "X" not in data.keys() or "Z" not in data.keys():
                raise ValueError(
                    "True latent positions must be provided when use_true_latent is True."
                )
            X = data["X"]
            Z = data["Z"]
            A = data.get("A", None)
            B = data.get("B", None)
            Xhat = X.copy()
            Zhat = Z.copy()
        else:
            if "estimated_X" not in data.keys():
                # need to estimate latent positions
                A = data.get("A", None)
                B = data.get("B", None)
                self.A = A
                self.B = B
                # true latent positions may not be provided
                X = data.get("X", None)
                Z = data.get("Z", None)

                # get the number of dimensions (k). If X or Z is provided, use its
                # shape (i.e. the "true" value of k)
                if X is not None or Z is not None:
                    self.k = X.shape[1] if X is not None else Z.shape[1]
                else:
                    if self.k is None:
                        raise ValueError(
                            "Number of dimensions (k) must be specified if X and Z are not provided."
                        )
                    self.k = self.k

                Zhat = self.solver(A, k=self.k, rng=self.rng)[
                    0
                ]  # 0 is the xhat, 1 are the evalues
                Xhat = self.solver(B, k=self.k, rng=self.rng)[0]

            else:
                Zhat = data.get("estimated_Z")
                Xhat = data.get("estimated_X")
                Z = data.get("Z", None)
                X = data.get("X", None)
                A = data.get("A", None)
                B = data.get("B", None)

        self.A = A
        self.B = B
        self.X = X
        self.Z = Z
        self.Zhat = Zhat
        self.Xhat = Xhat


class BasePermutationTest(BaseMethod):
    """Base class for permutation tests"""

    def __init__(
        self,
        k=None,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent_x=False,
        use_true_latent_z=False,
        test_function=None,
        permutation_type="latent",
        **kwargs,
    ):
        super().__init__()

        self.rng = rng if rng is not None else np.random.default_rng()

        if solver is None:
            raise ValueError("Solver must be provided")

        self.k = k
        self.solver = solver
        self.npermutations = npermutations
        self.alpha = alpha
        self.use_true_latent_x = use_true_latent_x
        self.use_true_latent_z = use_true_latent_z

        if test_function is None:
            raise ValueError("Test function must be provided")
        self.test_function = test_function

        if permutation_type not in ["latent", "observed"]:
            raise ValueError(
                "Invalid permutation_type. Must be 'latent' or 'observed'."
            )
        self.permutation_type = permutation_type

    def _fit_permutation(self):
        """Get pvalue using permutation test"""
        # A technically not needed but looks cleaner
        A, B, Zhat, Xhat = self.A, self.B, self.Zhat, self.Xhat
        
        self.permutation_distribution = []

        test_stat_estimate = self.test_function(Zhat, Xhat)
        self.test_stat_estimate = test_stat_estimate

        if self.permutation_type == "observed":
            for _ in range(self.npermutations):
                perm = self.rng.permutation(B.shape[0])
                # permute only one of the two
                B_perm = B[perm][:, perm]

                Xhat_perm = self.solver(B_perm, k=self.k, rng=self.rng)[0]
                test_stat_perm = self.test_function(Zhat, Xhat_perm)
                self.permutation_distribution.append(test_stat_perm)

        # estimate latent positions once, permute them and compute stat
        elif self.permutation_type == "latent":
            for _ in range(self.npermutations):
                perm = self.rng.permutation(Zhat.shape[0])
                Xhat_perm = Xhat[perm, :]
                test_stat_perm = self.test_function(Zhat, Xhat_perm)
                self.permutation_distribution.append(test_stat_perm)

        # get and store pvalue
        self.pvalue = np.mean(
            np.abs(self.permutation_distribution) >= np.abs(self.test_stat_estimate)
        )

        self.reject_null = bool(self.pvalue < self.alpha)

    def _process_input(self, data):
        """Utility function to extract and process input data, estimate latent
        positions if needed, and store results in the object."""

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary with keys 'A', 'B'."
            )

        # split between using true latent for x or z
        if self.use_true_latent_x is True:
            if "X" not in data.keys():
                raise ValueError(
                    "True latent positions for X must be provided when use_true_latent_x is True."
                )
            X = data["X"]
            Xhat = X.copy()
            B = data.get("B", None)
        else:
            if "estimated_X" not in data.keys():
                # estimate Xhat
                if "B" not in data.keys():
                    raise ValueError(
                        "Adjacency matrix B must be provided to estimate Xhat when use_true_latent_x is False."
                    )
                B = data["B"]
                # true latent positions may not be provided
                X = data.get("X", None)
                if self.k is None:
                    raise ValueError("Number of dimensions (k) must be specified")

                Xhat = self.solver(B, k=self.k, rng=self.rng)[0]
            else:
                Xhat = data.get("estimated_X")
                B = data.get("B", None)
                X = data.get("X", None)

        if self.use_true_latent_z is True:
            if "Z" not in data.keys():
                raise ValueError(
                    "True latent positions for Z must be provided when use_true_latent_z is True."
                )
            Z = data["Z"]
            Zhat = Z.copy()
            A = data.get("A", None)
        else:
            if "estimated_Z" not in data.keys():
                # need to estimate latent positions
                A = data.get("A", None)
                self.A = A
                # true latent positions may not be provided
                Z = data.get("Z", None)

                # get the number of dimensions (k). If X or Z is provided, use its
                # shape (i.e. the "true" value of k)
                if self.k is None:
                    raise ValueError("Number of dimensions (k) must be provided.")

                # 0 is the xhat, 1 are the eigenvalues
                Zhat = self.solver(A, k=self.k, rng=self.rng)[0]
            else:
                Zhat = data.get("estimated_Z")
                Z = data.get("Z", None)
                A = data.get("A", None)

        self.A = A
        self.B = B
        self.X = X
        self.Z = Z
        self.Zhat = Zhat
        self.Xhat = Xhat
