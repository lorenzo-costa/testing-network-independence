import numpy as np
from sklearn.model_selection import KFold

class ConditionalEstimator:
    """Interface for an estimated conditional law of Y given X."""

    def fit(self, X, Y):
        raise NotImplementedError

    def sample(self, X):
        raise NotImplementedError

    def pdf(self, X, Y):
        raise NotImplementedError

class GaussianKDEConditionalEstimator(ConditionalEstimator):
    """Estimate a conditional density with Gaussian product kernels.

    A shared bandwidth smooths every X coordinate. A list of bandwidths
    triggers cross-validation. A shared y_bandwidth smooths every Y coordinate;
    by default it uses a dimension-aware Scott-style rule. Each sample selects
    a whole training Y row before adding Gaussian noise.
    """

    def __init__(self, bandwidth=1.0, n_splits=5, y_bandwidth=None, rng=None):
        candidates = np.atleast_1d(np.asarray(bandwidth, dtype=float))
        if (candidates.ndim != 1 or candidates.size == 0
                or not np.all(np.isfinite(candidates)) or np.any(candidates <= 0)):
            raise ValueError("bandwidth must contain positive finite numbers.")
        if y_bandwidth is not None:
            if (np.ndim(y_bandwidth) != 0 or not np.isfinite(y_bandwidth)
                    or y_bandwidth <= 0):
                raise ValueError("y_bandwidth must be a positive finite scalar.")
        if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")
        self.bandwidth = bandwidth
        self.candidates = candidates
        self.n_splits = n_splits
        self.y_bandwidth = y_bandwidth
        self.rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def _matrix(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if (values.ndim != 2 or values.shape[1] < 1
                or not np.all(np.isfinite(values))):
            raise ValueError(f"{name} must be a finite matrix with at least one column.")
        return values

    @staticmethod
    def _default_y_bandwidth(y):
        scale = np.sqrt(np.mean(np.var(y, axis=0, ddof=1)))
        if not np.isfinite(scale) or scale == 0:
            scale = max(1.0, abs(float(np.mean(y))))
        return max(np.finfo(float).eps, scale * len(y) ** (-1 / (y.shape[1] + 4)))

    @staticmethod
    def _log_pdf(x_query, y_query, x_train, y_train, x_bw, y_bw):
        from scipy.special import logsumexp
        dx = (x_query[:, None, :] - x_train[None, :, :]) / x_bw
        dy = (y_query[:, None, :] - y_train[None, :, :]) / y_bw
        log_weights = -0.5 * np.sum(dx**2, axis=2)
        return (logsumexp(log_weights - 0.5 * np.sum(dy**2, axis=2), axis=1)
                - logsumexp(log_weights, axis=1)
                - y_query.shape[1] * np.log(y_bw * np.sqrt(2 * np.pi)))

    def fit(self, X, Y):
        x = self._matrix(X, "X")
        y = self._matrix(Y, "Y")
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("X and Y must have the same length of at least two.")
        self.x_train_ = x[:, 0].copy() if x.shape[1] == 1 else x.copy()
        self.y_train_ = y[:, 0].copy() if y.shape[1] == 1 else y.copy()
        self.x_dimension_ = x.shape[1]
        self.y_dimension_ = y.shape[1]
        self.y_bandwidth_ = (float(self.y_bandwidth) if self.y_bandwidth is not None
                             else self._default_y_bandwidth(y))
        if len(self.candidates) == 1:
            self.bandwidth_ = float(self.candidates[0])
        else:
            if self.n_splits > len(x):
                raise ValueError("n_splits cannot exceed the training sample size.")
            scores = []
            for bw in self.candidates:
                fold_scores = []
                for train, test in KFold(n_splits=self.n_splits, shuffle=True,
                                          random_state=0).split(x):
                    y_bw = (float(self.y_bandwidth) if self.y_bandwidth is not None
                            else self._default_y_bandwidth(y[train]))
                    fold_scores.extend(self._log_pdf(
                        x[test], y[test], x[train], y[train], bw, y_bw
                    ))
                scores.append(np.mean(fold_scores))
            self.bandwidth_ = float(self.candidates[np.argmax(scores)])
        return self

    def _check_fitted(self):
        if not hasattr(self, "x_train_"):
            raise RuntimeError("Fit the conditional estimator before using it.")

    def pdf(self, X, Y):
        """Evaluate one conditional density per paired query row."""
        self._check_fitted()
        x = self._matrix(X, "X")
        y = self._matrix(Y, "Y")
        if len(x) != len(y):
            raise ValueError("X and Y queries must have the same length.")
        if x.shape[1] != self.x_dimension_ or y.shape[1] != self.y_dimension_:
            raise ValueError("X and Y queries must match the fitted column counts.")
        return np.exp(self._log_pdf(
            x, y, self._matrix(self.x_train_, "X"),
            self._matrix(self.y_train_, "Y"), self.bandwidth_, self.y_bandwidth_
        ))

    def sample(self, X):
        """Draw one joint Y row per query X; return shape (n, y_dimension)."""
        from scipy.special import logsumexp
        self._check_fitted()
        x = self._matrix(X, "X")
        if x.shape[1] != self.x_dimension_:
            raise ValueError("X queries must match the fitted column count.")
        x_train = self._matrix(self.x_train_, "X")
        log_weights = -0.5 * np.sum(
            ((x[:, None, :] - x_train[None, :, :]) / self.bandwidth_) ** 2,
            axis=2,
        )
        weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        uniforms = self.rng.uniform(size=len(x))
        indices = (np.cumsum(weights, axis=1) < uniforms[:, None]).sum(axis=1)
        indices = np.minimum(indices, len(self.y_train_) - 1)
        y_train = self._matrix(self.y_train_, "Y")
        return y_train[indices] + self.rng.normal(
            scale=self.y_bandwidth_, size=(len(x), self.y_dimension_)
        )
