from ._base_class import BaseMethod
import numpy as np


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


class QAP(BaseMethod):
    """Global studentized QAP for symmetric dyadic networks.

    This implements the Section 4 global-null test with ``Ay`` as the outcome
    network and all matrices in ``Ax`` as dyadic regressors. The tested null is
    that every slope coefficient is zero. Only ``Ay`` is permuted.

    Parameters
    ----------
    alpha : float, optional
        Significance level (default 0.05).
    npermutations : int, optional
        Number of outcome-network permutations (default 100).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        alpha=0.05,
        npermutations=100,
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

        self.alpha = alpha
        self.npermutations = int(npermutations)
        self.rng = np.random.default_rng() if rng is None else rng
        self.permutation_distribution = []
        self.permutation_indices = []

        # BaseMethod.get_estimated compatibility.
        self.X = None
        self.Z = None
        self.Xhat = None
        self.Zhat = None

    def fit(self, data, **kwargs):
        """Test the global null that every Ax coefficient is zero.

        Parameters
        ----------
        data : dict
            DGP output containing the symmetric outcome network ``Ay`` and a
            non-empty list of symmetric covariate networks ``Ax``.
        """
        Ay, Ax = self._process_input(data)
        self.Ay = Ay
        self.Ax = Ax
        self.A = Ay
        self.B = Ax

        (
            self.test_stat_estimate,
            self.coefficients,
            self.coefficient_covariance,
        ) = self._compute_studentized_wald(Ay, Ax)

        self.permutation_distribution = []
        self.permutation_indices = []
        for _ in range(self.npermutations):
            permutation = self.rng.permutation(Ay.shape[0])
            self.permutation_indices.append(permutation)
            permuted_Ay = Ay[permutation][:, permutation]
            statistic, _, _ = self._compute_studentized_wald(permuted_Ay, Ax)
            self.permutation_distribution.append(statistic)

        null = np.asarray(self.permutation_distribution)
        extreme = np.count_nonzero(null >= self.test_stat_estimate)
        self.pvalue = (extreme + 1) / (self.npermutations + 1)

        self.reject_null = bool(self.pvalue < self.alpha)

    def get_name(self):
        return "QAP"

    @classmethod
    def _process_input(cls, data):
        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary containing Ay and Ax."
            )
        if data.get("Ay") is None:
            raise ValueError("Ay must be provided.")
        if data.get("Ax") is None:
            raise ValueError("Ax must be provided.")

        Ay = cls._validate_symmetric_network(data["Ay"], "Ay")
        raw_Ax = data["Ax"]
        if not isinstance(raw_Ax, (list, tuple)) or len(raw_Ax) == 0:
            raise ValueError("Ax must be a non-empty list or tuple of networks.")

        Ax = [
            cls._validate_symmetric_network(network, f"Ax[{index}]", Ay.shape[0])
            for index, network in enumerate(raw_Ax)
        ]
        return Ay, Ax

    @staticmethod
    def _validate_symmetric_network(network, name, n=None):
        network = np.asarray(network, dtype=float)
        if network.ndim != 2 or network.shape[0] != network.shape[1]:
            raise ValueError(f"{name} must be a square matrix.")
        if n is not None and network.shape != (n, n):
            raise ValueError(f"{name} must have shape ({n}, {n}); got {network.shape}.")
        if not np.isfinite(network).all():
            raise ValueError(f"{name} must contain only finite values.")
        if not np.allclose(network, network.T):
            raise ValueError(f"{name} must be symmetric.")
        return network

    @staticmethod
    def _require_invertible(matrix, name):
        tolerance = np.finfo(float).eps * max(matrix.shape) * np.linalg.norm(
            matrix, ord=2
        )
        if np.linalg.matrix_rank(matrix, tol=tolerance) < matrix.shape[0]:
            raise ValueError(f"{name} is singular; the global QAP is degenerate.")

    @classmethod
    def _compute_studentized_wald(cls, Ay, Ax):
        """Return the Section 4 Wald statistic, slopes, and their covariance."""
        n = Ay.shape[0]
        p = len(Ax)
        off_diagonal = ~np.eye(n, dtype=bool)
        dyad_count = n * (n - 1)

        outcome = Ay[off_diagonal]
        regressors = np.column_stack(
            [network[off_diagonal] for network in Ax]
        )
        centered_outcome = outcome - outcome.mean()
        centered_regressors = regressors - regressors.mean(axis=0)

        sigma_bb = centered_regressors.T @ centered_regressors / dyad_count
        cls._require_invertible(sigma_bb, "The dyadic regressor covariance")
        sigma_ba = centered_regressors.T @ centered_outcome / dyad_count
        coefficients = np.linalg.solve(sigma_bb, sigma_ba)

        residuals = centered_outcome - centered_regressors @ coefficients
        residual_matrix = np.zeros_like(Ay, dtype=float)
        residual_matrix[off_diagonal] = residuals

        centered_networks = np.zeros((n, n, p), dtype=float)
        regressor_means = regressors.mean(axis=0)
        for index, network in enumerate(Ax):
            centered_networks[:, :, index] = network - regressor_means[index]
            centered_networks[np.diag_indices(n)[0], np.diag_indices(n)[1], index] = 0

        projected_scores = np.einsum(
            "ij,ijp->ip", residual_matrix, centered_networks
        ) / (n - 1)
        h1_phi = projected_scores.T @ projected_scores / n

        sigma_inverse = np.linalg.inv(sigma_bb)
        coefficient_covariance = 4 * sigma_inverse @ h1_phi @ sigma_inverse
        coefficient_covariance = (
            coefficient_covariance + coefficient_covariance.T
        ) / 2
        cls._require_invertible(
            coefficient_covariance,
            "The studentizing covariance estimator",
        )

        statistic = n * coefficients @ np.linalg.solve(
            coefficient_covariance, coefficients
        )
        return float(max(statistic, 0.0)), coefficients, coefficient_covariance


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
