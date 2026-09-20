import os
from multiprocessing import Pool, current_process

import numpy as np
from tqdm.auto import tqdm


_PERMUTATION_WORKER_METHOD = None


def _initialize_permutation_worker(method):
    """Install one read-only method copy in each permutation worker."""
    global _PERMUTATION_WORKER_METHOD
    _PERMUTATION_WORKER_METHOD = method


def _run_permutation_worker(task):
    """Evaluate one indexed permutation in a process-pool worker."""
    return _PERMUTATION_WORKER_METHOD._evaluate_permutation(task)


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
                raise ValueError(
                    "True Z must be provided when use_true_latent is True."
                )
            Zhat = np.asarray(Z).copy()
        elif "estimated_Z" in data:
            Zhat = np.asarray(data["estimated_Z"])
        else:
            if A is None:
                raise ValueError(
                    "A must be provided to estimate Z when use_true_latent is False."
                )
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


class BasePermutationTest(BaseMethod):
    """Test network Y against the concatenated latent positions of p X networks.

    Parameters
    ----------
    d_y, d_x : int, optional
        Embedding dimension of Y and of each X network. When omitted, infer
        them from true latent arrays in the input dictionary, if available.
    solver : callable, optional
        Called for Y first, then each X network in order, as
        ``solver(adjacency, k=dimension, rng=rng)``. Its first return value
        must have shape ``(n, dimension)``.
    test_function : callable
        Receives ``(Yhat, Xhat)`` with shapes ``(n, d_y)`` and ``(n, p*d_x)``.
        May return a scalar or a one-dimensional vector of statistics.
    use_true_latent : bool, default=False
        Use supplied Y and X latent positions instead of fitting a solver.
    permutation_type : {"latent", "adjacency"}, default="latent"
        Permute estimated Y rows or permute Y's adjacency matrix and refit its
        embedding. All X embeddings stay fixed in either mode.
    npermutations : int, default=100
        Number of random permutations. P-values use the finite-sample correction.
    alpha : float, default=0.05
        Rejection threshold.
    one_sided : bool, default=False
        Use an upper-tail comparison for scalar statistics; otherwise compare
        absolute values. Vector statistics always use standardized maxima.
    n_jobs : int, default=1
        Number of worker processes; -1 selects all available CPUs.
    batch_size : int, default=32
        Target number of task batches per worker.
    verbose : bool, default=False
        Display permutation progress.
    rng : numpy.random.Generator, optional
        Source of randomness for embeddings and permutations.
    """

    def __init__(
        self,
        d_y=None,
        d_x=None,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent=False,
        test_function=None,
        permutation_type="latent",
        one_sided=False,
        n_jobs=1,
        batch_size=32,
        verbose=False,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.solver = solver
        self.use_true_latent = use_true_latent
        self.d_y = self._embedding_dimension(d_y, "d_y")
        self.d_x = self._embedding_dimension(d_x, "d_x")
        if not use_true_latent and solver is None:
            raise ValueError("Solver must be provided when use_true_latent is False")
        if test_function is None:
            raise ValueError("Test function must be provided")
        if permutation_type not in {"latent", "adjacency"}:
            raise ValueError(
                "Invalid permutation_type. Must be 'latent' or 'adjacency'."
            )
        if permutation_type == "adjacency" and use_true_latent:
            raise ValueError(
                "permutation_type='adjacency' requires use_true_latent=False."
            )
        if (
            isinstance(n_jobs, bool)
            or not isinstance(n_jobs, (int, np.integer))
            or (n_jobs != -1 and n_jobs < 1)
        ):
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        self.npermutations = npermutations
        self.alpha = alpha
        self.test_function = test_function
        self.permutation_type = permutation_type
        self.one_sided = bool(one_sided)
        self.n_jobs = int(n_jobs)
        self.batch_size = int(batch_size)
        self.verbose = bool(verbose)

    @staticmethod
    def _embedding_dimension(value, name):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
        return None if value is None else int(value)

    @staticmethod
    def _matrix(values, name):
        """Validate a finite, nonempty matrix without changing caller data."""
        try:
            if np.iscomplexobj(values):
                raise ValueError
            matrix = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite real 2D array.") from error
        if matrix.ndim != 2 or 0 in matrix.shape or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite, nonempty 2D array.")
        return matrix

    @classmethod
    def _matrix_list(cls, values, name):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name} must be a nonempty list of matrices.")
        return [cls._matrix(value, f"{name}[{i}]") for i, value in enumerate(values)]

    def _estimate_latent(self, adjacency, dimension, name, rng):
        estimated = self._matrix(
            self.solver(adjacency.copy(), k=dimension, rng=rng)[0], name
        )
        expected_shape = (adjacency.shape[0], dimension)
        if estimated.shape != expected_shape:
            raise ValueError(
                f"Solver output {name} must have shape {expected_shape}; got {estimated.shape}."
            )
        return estimated.copy()

    def _process_input(self, data):
        """Estimate Y and each X block from a DGP dictionary or ordered networks."""
        if isinstance(data, (list, tuple)):
            if len(data) < 2:
                raise ValueError(
                    "Provide at least two networks, ordered as [A_Y, A_X1, ...]."
                )
            data = {"A_Y": data[0], "A_X": list(data[1:])}
        if not isinstance(data, dict):
            raise ValueError(
                "Expected a dictionary with A_Y/A_X or an ordered list of networks."
            )

        true_y = self._matrix(data["Y"], "Y") if data.get("Y") is not None else None
        true_x = (
            self._matrix_list(data["X"], "X") if data.get("X") is not None else None
        )
        if true_x is not None and any(x.shape != true_x[0].shape for x in true_x):
            raise ValueError("All X latent blocks must have the same shape (n, d_x).")

        a_y, a_x = None, None
        if data.get("A_Y") is not None or data.get("A_X") is not None:
            if data.get("A_Y") is None or data.get("A_X") is None:
                raise ValueError("A_Y and A_X must be supplied together.")
            a_y = self._matrix(data["A_Y"], "A_Y")
            a_x = self._matrix_list(data["A_X"], "A_X")
            n = a_y.shape[0]
            if a_y.shape != (n, n):
                raise ValueError("A_Y must be square with shape (n, n).")
            if any(a.shape != (n, n) for a in a_x):
                raise ValueError(
                    "All A_X networks must have shape (n, n), matching A_Y."
                )
            p = len(a_x)
        elif self.use_true_latent and true_y is not None and true_x is not None:
            n, p = true_y.shape[0], len(true_x)
        else:
            raise ValueError(
                "Supply A_Y and A_X, or Y and X when use_true_latent=True."
            )

        if true_y is not None and true_y.shape[0] != n:
            raise ValueError("Y must have n rows, matching the networks.")
        if true_x is not None and (len(true_x) != p or true_x[0].shape[0] != n):
            raise ValueError("X must contain p latent blocks with n rows each.")

        d_y = (
            self.d_y
            if self.d_y is not None
            else (true_y.shape[1] if true_y is not None else None)
        )
        d_x = (
            self.d_x
            if self.d_x is not None
            else (true_x[0].shape[1] if true_x is not None else None)
        )
        if self.use_true_latent:
            if true_y is None or true_x is None:
                raise ValueError(
                    "Y and X must both be provided when use_true_latent=True."
                )
            if true_y.shape[1] != d_y or true_x[0].shape[1] != d_x:
                raise ValueError(
                    "True latent shapes must match d_y and d_x when use_true_latent=True."
                )
            yhat = true_y.copy()
            xhat_blocks = [x.copy() for x in true_x]
        else:
            if d_y is None or d_x is None:
                raise ValueError(
                    "Specify d_y and d_x when true latent dimensions are unavailable."
                )
            yhat = self._estimate_latent(a_y, d_y, "Yhat", self.rng)
            xhat_blocks = [
                self._estimate_latent(a, d_x, f"Xhat[{i}]", self.rng)
                for i, a in enumerate(a_x)
            ]

        self.n, self.p = n, p
        self.d_y, self.d_x = d_y, d_x
        self.A_Y, self.A_X = a_y, a_x
        self.Y, self.X = true_y, true_x
        self.Yhat = yhat
        self.Xhat_blocks = xhat_blocks
        self.Xhat = np.concatenate(xhat_blocks, axis=1)

    def fit(self, data):
        """Estimate the network latents and run the permutation test.

        Parameters
        ----------
        data : dict or sequence of arrays
            A DGP dictionary with A_Y of shape (n, n) and a list A_X of p
            networks of shape (n, n), or [A_Y, A_X1, ..., A_Xp]. Optional Y
            and X contain true latent positions with shapes (n, d_y) and a
            list of p arrays of shape (n, d_x). They are used directly only
            when use_true_latent=True.
        """
        self._process_input(data)
        self._fit_permutation()

    def get_estimated(self):
        """Return test matrices and individual X blocks for recovery metrics."""
        return {
            "estimated_latent": {
                "Y": self.Yhat,
                "X": self.Xhat,
                "X_blocks": self.Xhat_blocks,
            },
            "true_latent": {
                "Y": self.Y,
                "X": None if self.X is None else np.concatenate(self.X, axis=1),
                "X_blocks": self.X,
            },
            "p-value": self.pvalue,
            "reject_null": self.reject_null,
            "test_stat": self.test_stat_estimate,
        }

    def _draw_permutation(self):
        """Draw a node permutation for Y, keeping every X network fixed."""
        return self.rng.permutation(self.n)

    def _construct_permuted_data(self, permutation, rng):
        """Return permuted Y and the unchanged, concatenated X embeddings."""
        if self.permutation_type == "latent":
            return self.Yhat[permutation, :], self.Xhat

        if self.A_Y is None:
            raise ValueError("A_Y is required for adjacency permutations.")
        a_y_perm = self.A_Y[permutation][:, permutation]
        yhat_perm = self._estimate_latent(a_y_perm, self.d_y, "Yhat", rng)
        return yhat_perm, self.Xhat

    def _evaluate_test_statistic(self, Y, X, rng):
        """Evaluate a statistic, with a hook for RNG-aware subclasses."""
        return self.test_function(Y, X)

    @staticmethod
    def _as_statistic_array(statistic):
        """Return a scalar or one-dimensional statistic as a float array."""
        statistic = np.asarray(statistic, dtype=float)
        if statistic.ndim > 1:
            raise ValueError("Test statistics must be scalar or one-dimensional.")
        if statistic.size == 0:
            raise ValueError("Test statistics cannot be empty.")
        return statistic

    def _evaluate_permutation(self, task):
        """Evaluate one permutation task without mutating result state."""
        permutation_number, permutation, rng, expected_shape = task
        Y_stat, X_stat = self._construct_permuted_data(permutation, rng)
        statistic = self._as_statistic_array(
            self._evaluate_test_statistic(Y_stat, X_stat, rng)
        )
        if statistic.shape != expected_shape:
            raise ValueError(
                "Observed and permuted test statistics must have the same shape."
            )
        return permutation_number, statistic

    def _effective_n_jobs(self):
        """Resolve ``-1`` to all available CPUs."""
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return self.n_jobs

    def _collect_permutation_statistics(self, indexed_statistics):
        """Collect unordered results in permutation order and report progress."""
        ordered_statistics = [None] * self.npermutations
        with tqdm(
            total=self.npermutations,
            desc="Permutations",
            unit="permutation",
            disable=not self.verbose,
        ) as progress:
            for permutation_number, statistic in indexed_statistics:
                ordered_statistics[permutation_number] = statistic
                progress.update(1)

        return ordered_statistics

    def _compute_permutation_statistics(self):
        """Evaluate the observed and permuted scalar or vector statistics."""
        observed = self._as_statistic_array(
            self._evaluate_test_statistic(self.Yhat, self.Xhat, self.rng)
        )
        self.permutation_indices = [
            self._draw_permutation() for _ in range(self.npermutations)
        ]
        task_rngs = self.rng.spawn(self.npermutations)
        tasks = (
            (permutation_number, permutation, rng, observed.shape)
            for permutation_number, (permutation, rng) in enumerate(
                zip(self.permutation_indices, task_rngs)
            )
        )

        worker_count = min(self._effective_n_jobs(), max(1, self.npermutations))
        chunk_size = max(
            1,
            self.npermutations // (worker_count * self.batch_size),
        )
        self.permutation_chunk_size = chunk_size

        if worker_count == 1:
            statistics = map(self._evaluate_permutation, tasks)
            permuted_statistics = self._collect_permutation_statistics(statistics)
        else:
            if current_process().daemon:
                raise RuntimeError(
                    "Parallel permutation testing cannot run inside a parallel "
                    "simulation worker; set n_jobs=1 for either the permutation "
                    "test or the simulation."
                )
            with Pool(
                processes=worker_count,
                initializer=_initialize_permutation_worker,
                initargs=(self,),
            ) as pool:
                statistics = pool.imap_unordered(
                    _run_permutation_worker,
                    tasks,
                    chunksize=chunk_size,
                )
                permuted_statistics = self._collect_permutation_statistics(statistics)

        if observed.ndim == 0:
            permuted = np.asarray(permuted_statistics, dtype=float)
        elif permuted_statistics:
            permuted = np.stack(permuted_statistics)
        else:
            permuted = np.empty((0, observed.size), dtype=float)

        self.observed_statistics = observed
        self.permuted_statistics = permuted
        return observed, permuted

    def _compute_pvalue_and_rejection(self, observed, permuted):
        """Finalize scalar or max-standardized vector permutation statistics."""
        observed = self._as_statistic_array(observed)
        permuted = np.asarray(permuted, dtype=float)

        if observed.ndim == 0:
            if permuted.ndim != 1:
                raise ValueError("Permuted scalar statistics must be one-dimensional.")
            test_statistic = float(observed)
            null = permuted
        else:
            if permuted.ndim != 2 or permuted.shape[1:] != observed.shape:
                raise ValueError(
                    "Permuted vector statistics must have shape "
                    "(npermutations, nstatistics)."
                )
            if permuted.shape[0] < 1:
                raise ValueError(
                    "Vector-valued permutation tests require at least one "
                    "permutation for standardization."
                )

            statistic_matrix = np.vstack((observed, permuted))
            means = statistic_matrix.mean(axis=0)
            standard_deviations = statistic_matrix.std(axis=0, ddof=1)
            invalid_standard_deviation = ~np.isfinite(standard_deviations) | (
                standard_deviations == 0
            )
            if np.any(invalid_standard_deviation):
                indices = np.flatnonzero(invalid_standard_deviation).tolist()
                raise ValueError(
                    "Permutation statistic standard deviation is zero or "
                    f"non-finite for component indices {indices}."
                )

            standardized = (statistic_matrix - means) / standard_deviations
            max_statistics = standardized.max(axis=1)

            self.statistic_matrix = statistic_matrix
            self.statistic_means = means
            self.statistic_standard_deviations = standard_deviations
            self.standardized_statistics = standardized

            test_statistic = float(max_statistics[0])
            null = max_statistics[1:]

        if observed.ndim == 1 or self.one_sided:
            exceedances = np.count_nonzero(null >= test_statistic)
        else:
            exceedances = np.count_nonzero(np.abs(null) >= np.abs(test_statistic))

        self.test_stat_estimate = test_statistic
        self.permutation_distribution = null.tolist()
        self.pvalue = float((1 + exceedances) / (null.size + 1))
        self.reject_null = bool(self.pvalue < self.alpha)

    def _fit_permutation(self):
        observed, permuted = self._compute_permutation_statistics()
        self._compute_pvalue_and_rejection(observed, permuted)
