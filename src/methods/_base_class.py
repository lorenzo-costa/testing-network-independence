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
    """Evaluate one permutation in a process-pool worker."""
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
        one_sided=False,
        n_jobs=1,
        batch_size=32,
        verbose=False,
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
        self.stratify_permutations = bool(stratify_permutations)
        self.one_sided = bool(one_sided)
        self.n_jobs = int(n_jobs)
        self.batch_size = int(batch_size)
        self.verbose = bool(verbose)

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

    def _construct_permuted_data(self, permutation, rng):
        """Construct the latent positions and covariates for one permutation."""
        if self.permutation_type == "covariate":
            return self.Zhat, self.Y[permutation, :]
        if self.permutation_type == "latent":
            return self.Zhat[permutation, :], self.Y

        if self.A is None:
            raise ValueError("A is required for observed permutations.")
        A_perm = self.A[permutation][:, permutation]
        Zhat_perm = self.solver(A_perm, k=self.k, rng=rng)[0]
        return Zhat_perm, self.Y

    def _evaluate_test_statistic(self, Z, Y, rng):
        """Evaluate a statistic, with a hook for RNG-aware subclasses."""
        return self.test_function(Z, Y)

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
        permutation, rng, expected_shape = task
        Z_stat, Y_stat = self._construct_permuted_data(permutation, rng)
        statistic = self._as_statistic_array(
            self._evaluate_test_statistic(Z_stat, Y_stat, rng)
        )
        if statistic.shape != expected_shape:
            raise ValueError(
                "Observed and permuted test statistics must have the same shape."
            )
        return statistic

    def _effective_n_jobs(self):
        """Resolve ``-1`` to all available CPUs."""
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return self.n_jobs

    def _collect_permutation_statistics(self, statistics):
        """Collect statistics in permutation order and optionally show progress."""
        return list(
            tqdm(
                statistics,
                total=self.npermutations,
                desc="Permutations",
                unit="permutation",
                disable=not self.verbose,
            )
        )

    def _compute_permutation_statistics(self):
        """Evaluate the observed and permuted scalar or vector statistics."""
        observed = self._as_statistic_array(
            self._evaluate_test_statistic(self.Zhat, self.Y, self.rng)
        )
        self.permutation_indices = [
            self._draw_permutation() for _ in range(self.npermutations)
        ]
        task_rngs = self.rng.spawn(self.npermutations)
        tasks = (
            (permutation, rng, observed.shape)
            for permutation, rng in zip(self.permutation_indices, task_rngs)
        )

        worker_count = min(
            self._effective_n_jobs(), max(1, self.npermutations)
        )
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
                statistics = pool.imap(
                    _run_permutation_worker,
                    tasks,
                    chunksize=chunk_size,
                )
                permuted_statistics = self._collect_permutation_statistics(
                    statistics
                )

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
                raise ValueError(
                    "Permuted scalar statistics must be one-dimensional."
                )
            test_statistic = float(observed)
            null = permuted
            if self.one_sided:
                self.pvalue = float(np.mean(null >= test_statistic))
            else:
                self.pvalue = float(
                    np.mean(np.abs(null) >= np.abs(test_statistic))
                )
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
            invalid_standard_deviation = (
                ~np.isfinite(standard_deviations)
                | (standard_deviations == 0)
            )
            if np.any(invalid_standard_deviation):
                indices = np.flatnonzero(invalid_standard_deviation).tolist()
                raise ValueError(
                    "Permutation statistic standard deviation is zero or "
                    f"non-finite for component indices {indices}."
                )

            standardized = (
                statistic_matrix - means
            ) / standard_deviations
            max_statistics = standardized.max(axis=1)

            self.statistic_matrix = statistic_matrix
            self.statistic_means = means
            self.statistic_standard_deviations = standard_deviations
            self.standardized_statistics = standardized

            test_statistic = float(max_statistics[0])
            null = max_statistics[1:]
            exceedances = np.count_nonzero(null >= test_statistic)
            self.pvalue = float((1 + exceedances) / (null.size + 1))

        self.test_stat_estimate = test_statistic
        self.permutation_distribution = null.tolist()
        self.reject_null = bool(self.pvalue < self.alpha)

    def _fit_permutation(self):
        observed, permuted = self._compute_permutation_statistics()
        self._compute_pvalue_and_rejection(observed, permuted)
