from ._base_class import BaseMethod
import numpy as np


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class QAP(BaseMethod):
    """Quadratic Assignment Procedure

    Parameters
    ----------
    alpha : float, optional
        The significance level (default 0.05)
    npermutations : int, optional
        The number of permutations for the test (default 100)
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        alpha=0.05,
        npermutations=100,
        null_hypothesis="independence",
        rng=None,
        **args,
    ):
        super().__init__()

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.alpha = alpha
        self.npermutations = npermutations
        self.null_hypothesis = null_hypothesis
        self.permutation_distribution = []

        # for consistency with other methods
        self.X = None
        self.Z = None
        self.Xhat = None
        self.Zhat = None

    def fit(self, data, **kwargs):
        """Estimates the latent positions and computes p-value

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B' i.e. adjacency matrices
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary with keys 'A', 'B'."
            )

        A = data.get("A")
        B = data.get("B")
        self.A = A
        self.B = B
        n = A.shape[0]

        self.test_stat_estimate = self._compute_test_stat(A, B)
        self.permutation_distribution = []

        for i in range(self.npermutations):
            permutation = self.rng.permutation(n)
            B_perm = B[permutation, :][:, permutation]
            test_stat_perm = self._compute_test_stat(A, B_perm)
            self.permutation_distribution.append(test_stat_perm)

        # compute pvalue
        permutation_distribution = np.asarray(self.permutation_distribution)
        self.pvalue = (
            1
            + np.count_nonzero(
                np.abs(permutation_distribution) >= np.abs(self.test_stat_estimate)
            )
        ) / (self.npermutations + 1)

        self.reject_null = bool(self.pvalue < self.alpha)

        return

    def get_name(self):
        return "QAP"

    def _compute_test_stat(self, A, B):
        """Returns sqrt(n)rho if null hypothesis is independence (H0s) and sqrt(n)rho/v_w
        if null hypothesis is un-correlated (H0w). Test stats from SHi & Peng 2025"""

        n = A.shape[0]
        A_centered = A - A.mean(axis=0)
        B_centered = B - B.mean(axis=0)
        A_centered[np.arange(n), np.arange(n)] = 0
        B_centered[np.arange(n), np.arange(n)] = 0

        phi_0_hat = 1 / (n * (n - 1) - 1) * np.sum(A_centered * B_centered)

        eta_hat_2_alpha = 1 / (n * (n - 1) - 1) * np.sum(A_centered**2)
        eta_hat_2_beta = 1 / (n * (n - 1) - 1) * np.sum(B_centered**2)

        rho_hat = phi_0_hat / np.sqrt(eta_hat_2_alpha * eta_hat_2_beta)

        if self.null_hypothesis == "independence":
            return np.sqrt(n) * rho_hat

        eta_hat_1_phi = (
            1 / n * np.sum((1 / (n - 1) * np.sum(A_centered * B_centered, axis=1)) ** 2)
        )

        v_w_hat = 4 * eta_hat_1_phi / (eta_hat_2_alpha * eta_hat_2_beta)

        return np.sqrt(n) * rho_hat / np.sqrt(v_w_hat)


class MRQAP(BaseMethod):
    """Multiple Regression Quadratic Assignment Procedure.

    The tested network is ``A`` when it is supplied. Otherwise, ``Z`` is used,
    converting node-valued observations to pairwise Euclidean distances. The
    outcome ``Y`` and optional controls ``X`` may likewise be network- or
    node-valued.

    Parameters
    ----------
    alpha : float, optional
        Significance level (default 0.05).
    npermutations : int, optional
        Number of random node permutations (default 100).
    permutation_strategy : {"y_permutation", "dsp"}, optional
        Null permutation scheme. ``"dsp"`` implements double
        semipartialling (default ``"y_permutation"``).
    symmetric : bool, optional
        If True, regress on the upper-triangular dyads. If False, regress on
        all ordered off-diagonal dyads (default True).
    include_intercept : bool, optional
        Include an intercept in each regression (default True).
    batch_size : int or None, optional
        Number of permutations evaluated together. None evaluates one
        permutation at a time (default None).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    """

    _PERMUTATION_STRATEGIES = {"y_permutation", "dsp"}

    def __init__(
        self,
        alpha=0.05,
        npermutations=100,
        permutation_strategy="y_permutation",
        symmetric=True,
        include_intercept=True,
        batch_size=None,
        rng=None,
        **args,
    ):
        super().__init__()

        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")
        if (
            isinstance(npermutations, bool)
            or not isinstance(npermutations, (int, np.integer))
            or npermutations < 1
        ):
            raise ValueError("npermutations must be a positive integer.")
        if permutation_strategy not in self._PERMUTATION_STRATEGIES:
            raise ValueError("permutation_strategy must be 'y_permutation' or 'dsp'.")
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be None or a positive integer.")

        self.alpha = alpha
        self.npermutations = int(npermutations)
        self.permutation_strategy = permutation_strategy
        self.symmetric = bool(symmetric)
        self.include_intercept = bool(include_intercept)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.rng = np.random.default_rng() if rng is None else rng

        # BaseMethod.get_estimated compatibility
        self.X = None
        self.Z = None
        self.Zhat = None
        self.Xhat = None

    def fit(self, data, **kwargs):
        """Fit the observed regression and run the selected permutation test."""
        self._process_input(data)

        controls = self._vectorized_controls()
        control_design = self._control_design(controls)
        tested = self._vectorize(self.test_network)
        outcome = self._vectorize(self.outcome_network)
        design = np.column_stack((control_design, tested))

        q_full, r_full = self._full_rank_qr(
            design, "The MRQAP regression design is degenerate."
        )
        observed_coefficients = self._solve_qr(q_full, r_full, outcome)
        self.test_stat_estimate = float(observed_coefficients[-1])

        self.permutation_indices = [
            self.rng.permutation(self.n_nodes) for _ in range(self.npermutations)
        ]
        if self.permutation_strategy == "y_permutation":
            permutation_distribution = self._permute_y(q_full, r_full)
        else:
            permutation_distribution = self._double_semipartialling(
                control_design, tested, outcome
            )

        self.permutation_distribution = np.asarray(
            permutation_distribution, dtype=float
        )
        extreme = np.count_nonzero(
            np.abs(self.permutation_distribution) >= np.abs(self.test_stat_estimate)
        )
        self.pvalue = (extreme + 1) / (self.npermutations + 1)
        self.reject_null = bool(self.pvalue < self.alpha)

        return

    def get_name(self):
        return "MRQAP"

    def _process_input(self, data):
        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary containing Y and "
                "either A or Z, with optional X."
            )
        if data.get("Y") is None:
            raise ValueError("Y must be provided.")

        A = data.get("A")
        raw_z = data.get("Z")
        if A is not None:
            tested_network = np.asarray(A, dtype=float)
            if (
                tested_network.ndim != 2
                or tested_network.shape[0] != tested_network.shape[1]
            ):
                raise ValueError("A must have shape (n, n).")
            n = tested_network.shape[0]
        else:
            if raw_z is None:
                raise ValueError("Either A or Z must be provided.")
            tested_network, n = self._as_single_network(raw_z, "Z")

        outcome_network, outcome_n = self._as_single_network(data["Y"], "Y")
        if outcome_n != n:
            raise ValueError("A/Z and Y must contain the same number of nodes.")

        control_networks = self._as_control_networks(data.get("X"), n)
        self._require_finite(tested_network, "A/Z")
        self._require_finite(outcome_network, "Y")
        self._require_finite(control_networks, "X")

        self.n_nodes = n
        self.test_network = tested_network
        self.outcome_network = outcome_network
        self.control_networks = control_networks
        self.A = None if A is None else np.asarray(A)
        self.Y = np.asarray(data["Y"])
        self.X = None if data.get("X") is None else np.asarray(data["X"])
        self.Z = None if A is not None or raw_z is None else np.asarray(raw_z)

        if self.symmetric:
            self._row_indices, self._column_indices = np.triu_indices(n, k=1)
        else:
            self._row_indices, self._column_indices = np.where(~np.eye(n, dtype=bool))

    @staticmethod
    def _pairwise_distances(values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        differences = values[:, None, :] - values[None, :, :]
        return np.sqrt(np.sum(differences**2, axis=2))

    @classmethod
    def _as_single_network(cls, values, name):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            return cls._pairwise_distances(array), array.shape[0]
        if array.ndim != 2:
            raise ValueError(
                f"{name} must be node-valued with shape (n,) or (n, d), "
                "or network-valued with shape (n, n)."
            )
        if array.shape[0] == array.shape[1]:
            return array.copy(), array.shape[0]
        return cls._pairwise_distances(array), array.shape[0]

    @classmethod
    def _as_control_networks(cls, values, n):
        if values is None:
            return np.empty((n, n, 0), dtype=float)

        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            if array.shape[0] != n:
                raise ValueError("Node-valued X must contain n rows.")
            return cls._pairwise_distances(array)[:, :, None]

        if array.ndim == 2:
            if array.shape == (n, n):
                return array[:, :, None].copy()
            if array.shape[0] != n:
                raise ValueError("Node-valued X must contain n rows.")
            if array.shape[1] == 0:
                raise ValueError("Node-valued X must contain at least one column.")
            networks = [
                cls._pairwise_distances(array[:, j]) for j in range(array.shape[1])
            ]
            return np.stack(networks, axis=2)

        if array.ndim == 3 and array.shape[:2] == (n, n):
            return array.copy()

        raise ValueError(
            "X must be node-valued with shape (n,) or (n, k), or "
            "network-valued with shape (n, n) or (n, n, k)."
        )

    @staticmethod
    def _require_finite(values, name):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values.")

    def _vectorize(self, network):
        return np.asarray(network)[self._row_indices, self._column_indices]

    def _vectorize_permuted(self, network, permutation):
        return np.asarray(network)[
            permutation[self._row_indices], permutation[self._column_indices]
        ]

    def _vectorized_controls(self):
        if self.control_networks.shape[2] == 0:
            return np.empty((self._row_indices.size, 0), dtype=float)
        return self.control_networks[self._row_indices, self._column_indices, :]

    def _control_design(self, controls):
        if not self.include_intercept:
            return controls
        return np.column_stack((np.ones(controls.shape[0]), controls))

    @staticmethod
    def _full_rank_qr(design, error_message):
        if design.shape[1] == 0:
            return (
                np.empty((design.shape[0], 0), dtype=float),
                np.empty((0, 0), dtype=float),
            )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError(error_message)
        q, r = np.linalg.qr(design, mode="reduced")
        return q, r

    @staticmethod
    def _solve_qr(q, r, response):
        return np.linalg.solve(r, q.T @ response)

    def _permutation_batches(self):
        size = 1 if self.batch_size is None else self.batch_size
        for start in range(0, self.npermutations, size):
            yield self.permutation_indices[start : start + size]

    def _permute_y(self, q, r):
        coefficients = []
        for permutations in self._permutation_batches():
            responses = np.column_stack(
                [
                    self._vectorize_permuted(self.outcome_network, perm)
                    for perm in permutations
                ]
            )
            permuted_coefficients = self._solve_qr(q, r, responses)
            coefficients.extend(np.atleast_1d(permuted_coefficients[-1]).tolist())
        return coefficients

    def _double_semipartialling(self, control_design, tested, outcome):
        q_controls, _ = self._full_rank_qr(
            control_design, "The MRQAP control design is degenerate."
        )
        tested_residuals = self._residualize(q_controls, tested)
        outcome_residuals = self._residualize(q_controls, outcome)
        residual_network = self._reconstruct_network(tested_residuals)

        coefficients = []
        for permutations in self._permutation_batches():
            permuted_residuals = np.column_stack(
                [
                    self._vectorize_permuted(residual_network, perm)
                    for perm in permutations
                ]
            )
            u = self._residualize(q_controls, permuted_residuals)
            denominators = np.sum(u * u, axis=0)
            input_norms = np.sum(permuted_residuals * permuted_residuals, axis=0)
            tolerances = np.finfo(float).eps * max(1, u.shape[0]) * input_norms
            if np.any(denominators <= tolerances):
                raise ValueError(
                    "A permuted DSP regression is degenerate; the tested "
                    "network is unidentified after conditioning on X."
                )
            numerators = u.T @ outcome_residuals
            coefficients.extend((numerators / denominators).tolist())
        return coefficients

    @staticmethod
    def _residualize(q, values):
        return values - q @ (q.T @ values)

    def _reconstruct_network(self, values):
        network = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        network[self._row_indices, self._column_indices] = values
        if self.symmetric:
            network[self._column_indices, self._row_indices] = values
        return network
