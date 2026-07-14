import numpy as np


class BaseMethod:
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_estimated(self):
        """Return the common result structure for estimation and test methods."""
        return {
            "estimated_latent": self.Zhat,
            "true_latent": self.Z,
            "observed_Y": getattr(self, "Y", None),
            "conditioning_X": getattr(self, "X", None),
            "p-value": self.pvalue,
            "reject_null": self.reject_null,
            "test_stat": self.test_stat_estimate,
        }


class BaseEstimationMethod(BaseMethod):
    """Shared input processing for one-network methods."""

    def __init__(self, rng=None, solver=None, k=None, use_true_latent=False, **kwargs):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.solver = solver
        self.k = k
        self.use_true_latent = use_true_latent
        if not use_true_latent and solver is None:
            raise ValueError("Solver must be provided when use_true_latent is False")

    @staticmethod
    def _observed_y(data):
        if "Y" not in data:
            raise ValueError("Observed covariates Y must be provided.")
        Y = np.asarray(data["Y"])
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array with shape (n, ky).")
        return Y

    def _process_input(self, data):
        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary containing A, Z, Y, "
                "and optional X."
            )

        Y = self._observed_y(data)
        A = data.get("A")
        Z = data.get("Z")
        X = data.get("X")

        if self.use_true_latent:
            if Z is None:
                raise ValueError("True Z must be provided when use_true_latent is True.")
            Zhat = np.asarray(Z).copy()
        elif "estimated_Z" in data:
            Zhat = np.asarray(data["estimated_Z"])
        else:
            if A is None:
                raise ValueError("A must be provided to estimate Z when use_true_latent is False.")
            if self.solver is None:
                raise ValueError("Solver must be provided to estimate Z.")
            if self.k is None:
                if Z is None:
                    raise ValueError("Number of dimensions (k) must be specified.")
                self.k = np.asarray(Z).shape[1]
            Zhat = self.solver(A, k=self.k, rng=self.rng)[0]

        Zhat = np.asarray(Zhat)
        if Zhat.ndim != 2:
            raise ValueError("The tested latent representation must be a 2D array.")
        if self.k is None:
            self.k = Zhat.shape[1]
        if Zhat.shape[0] != Y.shape[0]:
            raise ValueError("A/Z and Y must contain the same number of nodes.")
        if A is not None and np.asarray(A).shape != (Y.shape[0], Y.shape[0]):
            raise ValueError("A must have shape (n, n), matching the rows of Y.")
        if X is not None:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.ndim != 2 or X.shape[0] != Y.shape[0]:
                raise ValueError("X must be a 2D array with n rows.")

        self.A = A
        self.Y = Y
        self.X = X
        self.Z = None if Z is None else np.asarray(Z)
        self.Zhat = Zhat


class BasePermutationTest(BaseEstimationMethod):
    """Permutation test for independence between network latent Z and observed Y."""

    def __init__(
        self,
        k=None,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent=False,
        test_function=None,
        permutation_type="covariate",
        stratify_permutations=False,
        **kwargs,
    ):
        super().__init__(
            rng=rng,
            solver=solver,
            k=k,
            use_true_latent=use_true_latent,
        )
        if test_function is None:
            raise ValueError("Test function must be provided")
        if permutation_type not in {"covariate", "latent", "observed"}:
            raise ValueError(
                "Invalid permutation_type. Must be 'covariate', 'latent', or 'observed'."
            )
        if permutation_type == "observed" and use_true_latent:
            raise ValueError(
                "permutation_type='observed' requires use_true_latent=False."
            )
        self.npermutations = npermutations
        self.alpha = alpha
        self.test_function = test_function
        self.permutation_type = permutation_type
        self.stratify_permutations = bool(stratify_permutations)

    def _draw_permutation(self):
        """Draw a global permutation or one restricted within rows of ``X``."""
        n = self.Y.shape[0]
        if not self.stratify_permutations or self.X is None:
            return self.rng.permutation(n)

        _, strata = np.unique(self.X, axis=0, return_inverse=True)
        permutation = np.arange(n)
        for stratum in np.unique(strata):
            indices = np.flatnonzero(strata == stratum)
            permutation[indices] = self.rng.permutation(indices)
        return permutation

    def _fit_permutation(self):
        self.test_stat_estimate = self.test_function(self.Zhat, self.Y)
        self.permutation_distribution = []
        self.permutation_indices = []

        for _ in range(self.npermutations):
            perm = self._draw_permutation()
            self.permutation_indices.append(perm.copy())
            if self.permutation_type == "covariate":
                statistic = self.test_function(self.Zhat, self.Y[perm, :])
            elif self.permutation_type == "latent":
                statistic = self.test_function(self.Zhat[perm, :], self.Y)
            else:
                if self.A is None:
                    raise ValueError("A is required for observed permutations.")
                A_perm = self.A[perm][:, perm]
                Zhat_perm = self.solver(A_perm, k=self.k, rng=self.rng)[0]
                statistic = self.test_function(Zhat_perm, self.Y)
            self.permutation_distribution.append(statistic)

        null = np.asarray(self.permutation_distribution)
        self.pvalue = np.mean(np.abs(null) >= np.abs(self.test_stat_estimate))
        self.reject_null = bool(self.pvalue < self.alpha)
