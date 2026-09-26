"""Validate and embed the ordered Y/X network inputs; assemble latent results."""

import numpy as np


class _MultipleNetworkInputMixin:
    """Shared Y-network/X-networks embedding and result handling."""

    def _initialize_multiple_network_input(
        self, *, rng, solver, d_y, d_x, use_true_latent
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.solver = solver
        self.use_true_latent = use_true_latent
        self.d_y = self._embedding_dimension(d_y, "d_y")
        self.d_x = self._embedding_dimension(d_x, "d_x")
        if not use_true_latent and solver is None:
            raise ValueError("Solver must be provided when use_true_latent is False")

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

    def get_estimated(self):
        """Return concatenated and per-network latent position estimates."""
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
