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
    """Estimate a scalar conditional density with Gaussian product kernels.

    bandwidth smooths in X. A list of bandwidths triggers cross-validation.
    y_bandwidth smooths in Y; by default it uses a Scott-style rule.
    """

    def __init__(self, bandwidth=1.0, n_splits=5, y_bandwidth=None, rng=None):
        candidates = np.atleast_1d(np.asarray(bandwidth, dtype=float))
        if (candidates.ndim != 1 or candidates.size == 0
                or not np.all(np.isfinite(candidates)) or np.any(candidates <= 0)):
            raise ValueError("bandwidth must contain positive finite numbers.")
        if y_bandwidth is not None and (not np.isfinite(y_bandwidth) or y_bandwidth <= 0):
            raise ValueError("y_bandwidth must be positive and finite.")
        if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")
        self.bandwidth = bandwidth
        self.candidates = candidates
        self.n_splits = n_splits
        self.y_bandwidth = y_bandwidth
        self.rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def _column(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite scalar column.")
        return values[:, 0]

    @staticmethod
    def _default_y_bandwidth(y):
        scale = np.std(y, ddof=1)
        if not np.isfinite(scale) or scale == 0:
            scale = max(1.0, abs(float(np.mean(y))))
        return max(np.finfo(float).eps, scale * len(y) ** (-1 / 5))

    @staticmethod
    def _log_pdf(x_query, y_query, x_train, y_train, x_bw, y_bw):
        from scipy.special import logsumexp
        dx = (x_query[:, None] - x_train[None, :]) / x_bw
        dy = (y_query[:, None] - y_train[None, :]) / y_bw
        log_weights = -0.5 * dx**2
        return (logsumexp(log_weights - 0.5 * dy**2, axis=1)
                - logsumexp(log_weights, axis=1)
                - np.log(y_bw * np.sqrt(2 * np.pi)))

    def fit(self, X, Y):
        x = self._column(X, "X")
        y = self._column(Y, "Y")
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("X and Y must have the same length of at least two.")
        self.x_train_ = x.copy()
        self.y_train_ = y.copy()
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
        x = self._column(X, "X")
        y = self._column(Y, "Y")
        if len(x) != len(y):
            raise ValueError("X and Y queries must have the same length.")
        return np.exp(self._log_pdf(
            x, y, self.x_train_, self.y_train_, self.bandwidth_, self.y_bandwidth_
        ))

    def sample(self, X):
        """Draw one conditional Y for each query X; return shape (n, 1)."""
        from scipy.special import logsumexp
        self._check_fitted()
        x = self._column(X, "X")
        log_weights = -0.5 * ((x[:, None] - self.x_train_[None, :]) / self.bandwidth_) ** 2
        weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        uniforms = self.rng.uniform(size=len(x))
        indices = (np.cumsum(weights, axis=1) < uniforms[:, None]).sum(axis=1)
        indices = np.minimum(indices, len(self.y_train_) - 1)
        draws = self.y_train_[indices] + self.rng.normal(scale=self.y_bandwidth_, size=len(x))
        return draws[:, None]

    def predict(self, X):
        """Return conditional draws for the original estimator interface."""
        return self.sample(X)
