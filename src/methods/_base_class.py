import numpy as np


class BaseMethod:
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_estimated(self):
        """Return the common result structure for estimation and test methods."""
        if hasattr(self, "Yhat"):
            estimated_latent = [self.Yhat, *self.Xhat_blocks]
            true_latent = None
            if self.Y is not None and self.X is not None:
                true_latent = [self.Y, *self.X]
        else:
            # Compatibility for methods such as QAP that do not estimate the
            # Y/X latent blocks.
            estimated_latent = getattr(self, "Yhat", None)
            true_latent = getattr(self, "Y", None)

        return {
            "estimated_latent": estimated_latent,
            "true_latent": true_latent,
            "observed_Y": getattr(self, "Y", None),
            "observed_X": getattr(self, "X", None),
            "p-value": self.pvalue,
            "reject_null": self.reject_null,
            "test_stat": self.test_stat_estimate,
        }


class BaseEstimationMethod(BaseMethod):
    """Shared input processing for tests between Y and all X networks."""

    def __init__(self, rng=None, solver=None, k=None, use_true_latent=False, **kwargs):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.solver = solver
        self.use_true_latent = bool(use_true_latent)

        if k is None:
            if not self.use_true_latent:
                raise ValueError(
                    "k must be provided when use_true_latent is False"
                )
            self.ky = None
            self.kx = None
        else:
            self.ky, self.kx = self._validate_dimensions(k)

        if not self.use_true_latent and solver is None:
            raise ValueError("Solver must be provided when use_true_latent is False")

    @staticmethod
    def _validate_dimensions(k):
        if not isinstance(k, (list, tuple)) or len(k) < 2:
            raise ValueError(
                "k must contain ky followed by at least one X dimension"
            )
        dimensions = []
        for index, dimension in enumerate(k):
            if isinstance(dimension, bool) or not isinstance(
                dimension, (int, np.integer)
            ):
                raise TypeError(f"k[{index}] must be an integer")
            if dimension <= 0:
                raise ValueError(f"k[{index}] must be positive")
            dimensions.append(int(dimension))
        return dimensions[0], tuple(dimensions[1:])

    @staticmethod
    def _validate_adjacency(matrix, name, n=None):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix")
        if n is not None and matrix.shape != (n, n):
            raise ValueError(f"{name} must have shape ({n}, {n}); got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must contain only finite values")
        return matrix

    @staticmethod
    def _validate_latent(matrix, name, n, dimension=None):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2 or matrix.shape[0] != n:
            raise ValueError(f"{name} must be a 2D array with {n} rows")
        if dimension is not None and matrix.shape[1] != dimension:
            raise ValueError(
                f"{name} must have shape ({n}, {dimension}); got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must contain only finite values")
        return matrix

    def _process_input(self, data):
        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary containing Ay, Ax, "
                "Y, and X."
            )

        if data.get("Ay") is None:
            raise ValueError("Adjacency matrix Ay must be provided.")
        if data.get("Ax") is None:
            raise ValueError("Adjacency matrices Ax must be provided.")

        Ay = self._validate_adjacency(data["Ay"], "Ay")
        n = Ay.shape[0]
        raw_Ax = data["Ax"]
        if not isinstance(raw_Ax, (list, tuple)) or len(raw_Ax) == 0:
            raise ValueError("Ax must contain at least one adjacency matrix")
        Ax = [
            self._validate_adjacency(matrix, f"Ax[{index}]", n=n)
            for index, matrix in enumerate(raw_Ax)
        ]

        Y = data.get("Y")
        X = data.get("X")
        if Y is not None:
            Y = self._validate_latent(Y, "Y", n)
        if X is not None:
            if not isinstance(X, (list, tuple)) or len(X) != len(Ax):
                raise ValueError(
                    "X must contain one latent-position array per Ax matrix"
                )
            X = [
                self._validate_latent(block, f"X[{index}]", n)
                for index, block in enumerate(X)
            ]

        if self.use_true_latent:
            if Y is None or X is None:
                raise ValueError(
                    "Y and X must be provided when use_true_latent is True"
                )
            if self.ky is not None and Y.shape[1] != self.ky:
                raise ValueError(
                    f"Y must have {self.ky} columns; got {Y.shape[1]}"
                )
            if self.kx is not None:
                if len(self.kx) != len(X):
                    raise ValueError(
                        f"k defines {len(self.kx)} X blocks, but data contains {len(X)}"
                    )
                for index, (block, dimension) in enumerate(zip(X, self.kx)):
                    if block.shape[1] != dimension:
                        raise ValueError(
                            f"X[{index}] must have {dimension} columns; "
                            f"got {block.shape[1]}"
                        )
            Yhat = Y.copy()
            Xhat_blocks = [block.copy() for block in X]
        else:
            if len(self.kx) != len(Ax):
                raise ValueError(
                    f"k defines {len(self.kx)} X blocks, but Ax contains {len(Ax)}"
                )
            Yhat = np.asarray(
                self.solver(Ay, k=self.ky, rng=self.rng)[0], dtype=float
            )
            Xhat_blocks = [
                np.asarray(
                    self.solver(matrix, k=dimension, rng=self.rng)[0],
                    dtype=float,
                )
                for matrix, dimension in zip(Ax, self.kx)
            ]

        Yhat = self._validate_latent(Yhat, "Yhat", n, self.ky)
        Xhat_blocks = [
            self._validate_latent(
                block,
                f"Xhat[{index}]",
                n,
                None if self.kx is None else self.kx[index],
            )
            for index, block in enumerate(Xhat_blocks)
        ]

        self.Ay = Ay
        self.Ax = Ax
        self.Yhat = Yhat
        self.Xhat_blocks = Xhat_blocks
        self.Xhat = np.concatenate(Xhat_blocks, axis=1)
        self.Y = Y
        self.X = X


class BasePermutationTest(BaseEstimationMethod):
    """Permutation test for global independence of Y and all X blocks."""

    def __init__(
        self,
        k=None,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent=False,
        test_function=None,
        permutation_type="latent",
        stratify_permutations=False,
        one_sided=False,
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
        if permutation_type not in {"latent", "observed"}:
            raise ValueError(
                "Invalid permutation_type. Must be 'latent' or 'observed'."
            )
        if permutation_type == "observed" and use_true_latent:
            raise ValueError(
                "permutation_type='observed' requires use_true_latent=False."
            )
        if isinstance(npermutations, bool) or not isinstance(
            npermutations, (int, np.integer)
        ) or npermutations < 1:
            raise ValueError("npermutations must be a positive integer")

        self.npermutations = int(npermutations)
        self.alpha = alpha
        self.test_function = test_function
        self.permutation_type = permutation_type
        self.stratify_permutations = bool(stratify_permutations)
        self.one_sided = bool(one_sided)

    def _fit_permutation(self):
        self.test_stat_estimate = self.test_function(self.Yhat, self.Xhat)
        self.permutation_distribution = []
        self.permutation_indices = []

        for _ in range(self.npermutations):
            permutation = self.rng.permutation(self.Yhat.shape[0])
            self.permutation_indices.append(permutation)

            if self.permutation_type == "observed":
                permuted_Ay = self.Ay[permutation][:, permutation]
                permuted_Yhat = self.solver(
                    permuted_Ay, k=self.ky, rng=self.rng
                )[0]
                statistic = self.test_function(permuted_Yhat, self.Xhat)
            else:
                statistic = self.test_function(
                    self.Yhat[permutation, :], self.Xhat
                )

            self.permutation_distribution.append(statistic)

        null = np.asarray(self.permutation_distribution)
        if self.one_sided:
            extreme = np.count_nonzero(null >= self.test_stat_estimate)
        else:
            extreme = np.count_nonzero(
                np.abs(null) >= np.abs(self.test_stat_estimate)
            )
        self.pvalue = (extreme + 1) / (self.npermutations + 1)
        self.reject_null = bool(self.pvalue < self.alpha)
