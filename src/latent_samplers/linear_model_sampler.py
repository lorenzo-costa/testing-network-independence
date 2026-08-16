
import numpy as np
from scipy import stats


class LinearModelGenerator:
    """Sampler for a linear latent-position data-generating process.

    The model is

        Y_i = sum_{j=1}^p B^{(j)} X_i^{(j)} + epsilon_i,

    where

        X_i^{(j)} in R^{d_x},
        Y_i in R^{d_y},
        B^{(j)} in R^{d_y x d_x},

    and epsilon is sampled independently of every X block.

    Parameters
    ----------
    n : int
        Number of samples (nodes).
    k : list of int
        first element is dimension of Y, the rest are dimensions of X blocks.
    B : array-like, optional
        Linear coefficient matrices. If None, one matrix per predictor block is
        sampled from a standard normal distribution. If ``"zero"``, zero
        matrices are used. Explicit coefficients may be supplied as a list in
        which element j has shape ``(d_y, d_x[j])``.
    marginals : dict or str, optional
        Scalar marginal distributions used to generate X and epsilon.
        If a string is supplied, it is used for both X and epsilon.
        If a dict is supplied, use keys ``"x"`` and ``"epsilon"``.
        ``marginals["x"]`` can itself be a single specification shared by
        every X block or a list of length p, one specification per block.
    noise_scale : float, default=1.0
        Multiplicative scale applied to epsilon after sampling.
    center_latent : bool, default=False
        If True, each X coordinate and each epsilon coordinate are centered
        by their sample means before Y is formed. This keeps the linear
        identity exact for the returned sample, but sample-centering makes
        rows no longer literally iid. Leave False for the exact iid DGP.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        n,
        k,
        B=None,
        marginals=None,
        noise_scale=1.0,
        center_latent=False,
        rng=None,
        **kwargs,
    ):
        self._validate_args(n=n, k=k, noise_scale=noise_scale)
        self.n = int(n)
        self.ky = int(k[0])
        self.kx = tuple(int(kj) for kj in k[1:])
        self.p = len(self.kx)  # Number of predictor blocks

        self.noise_scale = float(noise_scale)
        self.center_latent = bool(center_latent)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Coefficients are sampled once and retained for every latent draw.
        self.B = self._sample_coefficients(B)
        self._convert_marginals(marginals)

        self.is_null = all(np.allclose(Bj, 0.0) for Bj in self.B)

    def _coerce_B(self, B):
        """Validate explicit coefficients and return one matrix per X block."""
        if isinstance(B, np.ndarray):
            if self.p == 1 and B.shape == (self.ky, self.kx[0]):
                coefficient_matrices = [B]
            elif len(set(self.kx)) == 1 and B.shape == (
                self.p,
                self.ky,
                self.kx[0],
            ):
                coefficient_matrices = list(B)
            elif (
                self.ky == 1
                and len(set(self.kx)) == 1
                and B.shape == (self.p, self.kx[0])
            ):
                coefficient_matrices = [row[None, :] for row in B]
            else:
                coefficient_matrices = list(B)
        else:
            try:
                coefficient_matrices = list(B)
            except TypeError as exc:
                raise TypeError(
                    "B must be None, 'zero', or a sequence of coefficient matrices"
                ) from exc

        if len(coefficient_matrices) != self.p:
            raise ValueError(
                f"B must contain one matrix per predictor block ({self.p} total); "
                f"got {len(coefficient_matrices)}"
            )

        coefficients = []
        for j, (Bj, kxj) in enumerate(zip(coefficient_matrices, self.kx)):
            Bj = np.asarray(Bj, dtype=float)
            expected_shape = (self.ky, kxj)
            if Bj.shape != expected_shape:
                raise ValueError(
                    f"B[{j}] must have shape {expected_shape}; got {Bj.shape}"
                )
            if not np.isfinite(Bj).all():
                raise ValueError(f"B[{j}] must contain only finite values")
            coefficients.append(Bj.copy())

        return coefficients

    def _validate_args(self, n, k, noise_scale):
        if n <= 0:
            raise ValueError("n must be positive")
        if not isinstance(k, (list, tuple)) or len(k) < 2:
            raise ValueError("k must be a list or tuple of length at least 2")
        for i, ki in enumerate(k):
            if ki <= 0:
                raise ValueError(f"k[{i}] (= {ki}) must be positive")
        if noise_scale < 0:
            raise ValueError("noise_scale must be nonnegative")

    @staticmethod
    def _parse_dist(dist_spec):
        """Parse a scalar scipy distribution specification."""
        parts = dist_spec.split()
        name = parts[0]
        args = [float(p) for p in parts[1:]]

        registry = {
            "gaussian": (stats.norm, []),
            "exponential": (stats.expon, []),
            "uniform": (stats.uniform, []),
            "t": (stats.t, ["df"]),
            "chi": (stats.chi2, ["df"]),
            "chi2": (stats.chi2, ["df"]),
            "beta": (stats.beta, []),
            "gamma": (stats.gamma, ["a", "scale"]),
            "lognormal": (stats.lognorm, ["s"]),
            "cauchy": (stats.cauchy, ["loc", "scale"]),
            "dirichlet": (stats.dirichlet, ["alpha"]),
        }

        if name not in registry:
            raise ValueError(f"Unknown distribution: {name}")

        func, arg_keys = registry[name]

        if not args:
            return func
        if name == "uniform" and len(args) == 2:
            a, b = args
            return func(loc=a, scale=b - a)
        if name == "gamma" and len(args) == 2:
            return func(a=args[0], scale=args[1])
        if name == "cauchy" and len(args) == 2:
            return func(loc=args[0], scale=args[1])

        kwargs = {key: value for key, value in zip(arg_keys, args) if key}
        return func(**kwargs) if kwargs else func(*args)

    def _convert_marginals(self, marginals):
        """Normalize marginal specifications for X blocks and epsilon."""
        if marginals is None:
            marginals = {"x": "gaussian", "epsilon": "gaussian"}
        elif not isinstance(marginals, dict):
            marginals = {"x": marginals, "epsilon": marginals}

        if "x" not in marginals or "epsilon" not in marginals:
            raise ValueError('marginals must contain keys "x" and "epsilon"')

        x_specs = marginals["x"]
        if isinstance(x_specs, (list, tuple)):
            if len(x_specs) != self.p:
                raise ValueError(
                    f'If marginals["x"] is a list, it must have length p={self.p}'
                )
        else:
            x_specs = [x_specs] * self.p

        self.marginal_x = [self._parse_dist(spec) for spec in x_specs]
        self.marginal_epsilon = self._parse_dist(marginals["epsilon"])

    def _sample_predictors(self):
        """Sample one predictor array of shape (n, kx[j]) per block."""
        blocks = []

        for i in range(len(self.marginal_x)):
            dist = self.marginal_x[i]
            bb = dist.rvs(size=(self.n, self.kx[i]), random_state=self.rng)
            blocks.append(bb)

        for j, block in enumerate(blocks):
            if not np.isfinite(block).all():
                raise FloatingPointError(
                    f"Non-finite value encountered while sampling X block {j}"
                )
            if self.center_latent:
                blocks[j] = block - block.mean(axis=0, keepdims=True)

        return blocks

    def _sample_epsilon(self):
        """Sample epsilon independently of X with shape (n, d_y)."""
        epsilon = self.marginal_epsilon.rvs(
            size=(self.n, self.ky),
            random_state=self.rng,
        )
        epsilon = self.noise_scale * np.asarray(epsilon, dtype=float)

        if not np.isfinite(epsilon).all():
            raise FloatingPointError("Non-finite value encountered while sampling epsilon")

        if self.center_latent:
            epsilon = epsilon - epsilon.mean(axis=0, keepdims=True)

        return epsilon

    def _sample_coefficients(self, B):
        """Create coefficient matrices from a named strategy or explicit values.

        Add future sampling strategies to ``coefficient_samplers``. Each strategy
        must return a list whose j-th matrix has shape ``(ky, kx[j])``.
        """
        coefficient_samplers = {
            "gaussian": lambda: [
                self.rng.normal(loc=0.0, scale=1.0, size=(self.ky, kxj))
                for kxj in self.kx
            ],
            "zero": lambda: [
                np.zeros((self.ky, kxj), dtype=float) for kxj in self.kx
            ],
        }

        if B is None:
            strategy = "gaussian"
        elif isinstance(B, str):
            strategy = B
        else:
            return self._coerce_B(B)

        if strategy not in coefficient_samplers:
            available = ", ".join(sorted(coefficient_samplers))
            raise ValueError(
                f"Unknown coefficient sampling strategy: {strategy}. "
                f"Available strategies: {available}"
            )

        return coefficient_samplers[strategy]()

    def _sample_latent_linear(self, return_noise=False):
        """Sample latent positions from the linear model.

        Returns
        -------
        Y : ndarray, shape (n, d_y)
            Response latent positions.
        X : list of ndarray
            Predictor blocks. Element j has shape ``(n, kx[j])``.

        """
        X = self._sample_predictors()
        epsilon = self._sample_epsilon()

        # For each i:
        #   Y_i = sum_j B[j] @ X[i, j] + epsilon_i.
        signal = sum(Xj @ Bj.T for Xj, Bj in zip(X, self.B))
        Y = signal + epsilon

        if return_noise:
            return Y, X, epsilon
        return Y, X

    def get_name(self):
        return (
            f"linear_p{self.p}_dx{'x'.join(map(str, self.kx))}_dy{self.ky}_"
            f"noise{self.noise_scale:g}_null{int(self.is_null)}"
        )
