"""Public method contracts and permutation-test orchestration."""

from ._network_input import _MultipleNetworkInputMixin

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
            "p-value": self.pvalue,
            "reject_null": self.reject_null,
            "test_stat": self.test_stat_estimate,
        }


class BaseEstimationMethod(_MultipleNetworkInputMixin, BaseMethod):
    """Estimate Y and p X network latent positions without inference."""

    def __init__(
        self,
        rng=None,
        solver=None,
        d_y=None,
        d_x=None,
        use_true_latent=False,
        **kwargs,
    ):
        self._initialize_multiple_network_input(
            rng=rng,
            solver=solver,
            d_y=d_y,
            d_x=d_x,
            use_true_latent=use_true_latent,
        )


class BasePermutationTest(_MultipleNetworkInputMixin, BaseMethod):
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
        self._initialize_multiple_network_input(
            rng=rng,
            solver=solver,
            d_y=d_y,
            d_x=d_x,
            use_true_latent=use_true_latent,
        )
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
