"""Covariate-aware stochastic block model sampler.

The public class name ``SBM_covarite_generator`` intentionally preserves the
spelling requested by the original API.  ``SBMCovariateGenerator`` is supplied
as a PEP-8-friendly alias.
"""

from __future__ import annotations

import numpy as np



class SBMCovariateGenerator:
    """Sample an SBM together with a univariate node covariate ``Y``.

    Parameters
    ----------
    n
        Number of nodes.
    k
        Number of SBM communities.
    y_distribution
        Continuous covariate specification. It is a single whitespace-separated
        string. Supported forms are:

        - ``"gaussian [loc scale]"`` (``normal`` is accepted as an alias)
        - ``"cauchy [loc scale]"``
        - ``"uniform low high"``
        - ``"pareto shape [scale]"``; support is ``[scale, inf)``
        - ``"t df [loc scale]"``

        Bracketed values are optional only where stated. For example,
        ``"uniform 0 1"`` and ``"gaussian 0 1"`` are valid specifications.
        Set this argument to ``None`` for a discrete covariate.
    y_upper_bound
        Inclusive upper bound for a discrete covariate. A discrete covariate is
        sampled from ``{0, ..., y_upper_bound}``. Required when
        ``y_distribution`` is ``None``.
    y_probabilities
        Optional probability vector for values ``0, ..., y_upper_bound``.
        When omitted, the discrete covariate is sampled uniformly.
    sampling
        Relationship between Y and SBM labels Z:

        - ``"random"``: Z is independent of Y.
        - ``"identical"``: for discrete Y, use ``Z = Y``.
        - ``"correlated"``: for discrete Y, retain ``Z = Y`` with probability
          ``rho`` and otherwise switch to a uniformly selected different label.
        - ``"step"``: for continuous Y, partition the observed range of Y into
          k equal-width intervals and use the interval index as Z.
        - ``"softmax"``: for continuous Y, sample Z using
          ``P(Z=j | Y=y) ∝ exp(a_j + b_j y)``.
    rho
        Probability of retaining the same discrete label under
        ``sampling="correlated"``. Must lie in [0, 1].
    softmax_intercept, softmax_slope
        Scalars or length-k arrays giving ``a`` and ``b`` in the softmax model.
        Defaults are zero intercepts and slopes evenly spaced from -1 to 1.
    block_probs
        Optional user-supplied k-by-k block connection-probability matrix. When
        omitted, a random matrix is sampled using ``assortativity`` and
        ``sparsity_bias``.
    assortativity
        Number in [0, 1]. Values above 0.5 favor within-community edges; values
        below 0.5 favor between-community edges.
    sparsity_bias
        Number in [0, 1]. Larger values lower all sampled edge probabilities.
    symmetric
        Whether to sample a symmetric graph. Asymmetric graphs are directed and
        have no self-loops.
    self_loops
        Whether to permit self-loops. Ignored for symmetric graphs unless set
        to True.
    rng
        Optional NumPy ``Generator`` or seed.
    """

    _CONTINUOUS_DISTRIBUTIONS = {
        "gaussian",
        "normal",
        "cauchy",
        "uniform",
        "pareto",
        "t",
    }
    _SAMPLING_MODES = {"random", "identical", "correlated", "step", "softmax"}

    def __init__(
        self,
        n: int,
        k: int,
        ky: int = 1,
        *,
        y_distribution = None,
        y_upper_bound = None,
        y_probabilities = None,
        sampling = "random",
        rho = 0.8,
        softmax_intercept = None,
        softmax_slope = None,
        block_probs = None,
        assortativity = 0.5,
        sparsity_bias = 0.6,
        symmetric = False,
        self_loops = False,
        rng = None,
        **kwargs,
    ) -> None:
        self.n = n
        self.k = k
        if ky != 1:
            raise ValueError("SBMCovariateGenerator only supports ky=1.")
        self.ky = ky

        self.y_distribution = y_distribution
        self.y_upper_bound = y_upper_bound
        self.y_probabilities = y_probabilities
        self.sampling = str(sampling).lower()
        self.rho = float(rho)
        self.softmax_intercept = softmax_intercept
        self.softmax_slope = softmax_slope
        self.block_probs = block_probs
        self.assortativity = float(assortativity)
        self.sparsity_bias = float(sparsity_bias)
        self.symmetric = bool(symmetric)
        self.self_loops = bool(self_loops)
        self.rng = rng if rng is not None else np.random.default_rng()

        self._validate_parameters()
        self.covariate_type = (
            "continuous" if self.y_distribution is not None else "discrete"
        )

        # Convenience state populated by ``sample``.
        self.Y = None
        self.z = None
        self.community_assignment = None
        self.sampled_block_probs = None
        self.adjacency = None
        self.is_null = (self.sampling == "random")

    def _validate_parameters(self) -> None:
        if self.sampling not in self._SAMPLING_MODES:
            options = ", ".join(sorted(self._SAMPLING_MODES))
            raise ValueError(f"sampling must be one of {{{options}}}; got {self.sampling!r}.")

        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1].")
        if not -1.0 <= self.assortativity <= 1.0:
            raise ValueError("assortativity must lie in [-1, 1].")
        if not 0.0 <= self.sparsity_bias <= 1.0:
            raise ValueError("sparsity_bias must lie in [0, 1].")

        if self.y_distribution is not None and self.y_upper_bound is not None:
            raise ValueError(
                "Specify either y_distribution (continuous Y) or "
                "y_upper_bound (discrete Y), not both."
            )
        if self.y_distribution is None and self.y_upper_bound is None:
            raise ValueError(
                "Specify y_distribution for continuous Y or y_upper_bound "
                "for discrete Y."
            )

        if self.y_distribution is not None:
            self._parse_continuous_distribution(self.y_distribution)
            if self.y_probabilities is not None:
                raise ValueError(
                    "y_probabilities is only valid for a discrete covariate."
                )
            if self.sampling in {"identical", "correlated"}:
                raise ValueError(
                    f'sampling="{self.sampling}" requires a discrete covariate '
                    "(use y_upper_bound)."
                )
        else:
            if not isinstance(self.y_upper_bound, int) or self.y_upper_bound < 0:
                raise ValueError(
                    "y_upper_bound must be a nonnegative integer for a discrete covariate."
                )
            if self.sampling in {"step", "softmax"}:
                raise ValueError(
                    f'sampling="{self.sampling}" requires a continuous covariate '
                    "(use y_distribution)."
                )
            if self.sampling in {"identical", "correlated"} and self.k != (
                self.y_upper_bound + 1
            ):
                raise ValueError(
                    'For sampling="identical" or "correlated", k must equal '
                    "y_upper_bound + 1 so every Y value is a valid community "
                    "label."
                )
            if self.y_probabilities is not None:
                probs = np.asarray(self.y_probabilities, dtype=float)
                if probs.ndim != 1 or probs.size != self.y_upper_bound + 1:
                    raise ValueError(
                        "y_probabilities must be a 1D vector of length "
                        "y_upper_bound + 1."
                    )
                if np.any(probs < 0) or not np.isfinite(probs).all():
                    raise ValueError(
                        "y_probabilities must contain finite, nonnegative values."
                    )
                if not np.isclose(probs.sum(), 1.0):
                    raise ValueError("y_probabilities must sum to 1.")
                self.y_probabilities = probs

        if self.block_probs is not None:
            block_probs = np.asarray(self.block_probs, dtype=float)
            if block_probs.shape != (self.k, self.k):
                raise ValueError(
                    f"block_probs must have shape ({self.k}, {self.k}); got "
                    f"{block_probs.shape}."
                )
            if not np.isfinite(block_probs).all() or np.any(
                (block_probs < 0.0) | (block_probs > 1.0)
            ):
                raise ValueError("block_probs entries must lie in [0, 1].")
            if not self.symmetric and not np.allclose(block_probs, block_probs.T):
                raise ValueError(
                    "An undirected graph requires a symmetric block_probs matrix."
                )
            self.block_probs = block_probs

    @classmethod
    def _parse_continuous_distribution(
        cls, specification: str
    ) -> tuple[str, np.ndarray]:
        if not isinstance(specification, str) or not specification.strip():
            raise ValueError("y_distribution must be a non-empty string.")

        tokens = specification.split()
        name = tokens[0].lower()
        if name not in cls._CONTINUOUS_DISTRIBUTIONS:
            choices = ", ".join(sorted(cls._CONTINUOUS_DISTRIBUTIONS))
            raise ValueError(
                f"Unknown continuous distribution {name!r}. Choose from: {choices}."
            )

        try:
            params = np.asarray([float(value) for value in tokens[1:]], dtype=float)
        except ValueError as exc:
            raise ValueError(
                "Distribution hyperparameters must be numeric values in the "
                "y_distribution string."
            ) from exc

        expected = {
            "gaussian": (0, 2),
            "normal": (0, 2),
            "cauchy": (0, 2),
            "uniform": (0, 2),
            "pareto": (0, 1, 2),
            "t": (0, 1, 3),
        }[name]
        if params.size not in expected:
            examples = {
                "gaussian": '"gaussian 0 1"',
                "normal": '"normal 0 1"',
                "cauchy": '"cauchy 0 1"',
                "uniform": '"uniform 0 1"',
                "pareto": '"pareto 2 1"',
                "t": '"t 5 0 1"',
            }
            raise ValueError(
                f"Invalid hyperparameters for {name!r}. For example, use "
                f"{examples[name]}."
            )

        if name in {"gaussian", "normal", "cauchy"} and params.size == 2:
            if params[1] <= 0:
                raise ValueError(f"The scale for {name!r} must be positive.")
        elif name == "uniform":
            if params.size == 2:
                if not params[0] < params[1]:
                    raise ValueError(
                        "For uniform, the lower endpoint must be smaller than the "
                        "upper endpoint."
                    )
        elif name == "pareto":
            if params.size > 0:
                if params[0] <= 0 or (params.size == 2 and params[1] <= 0):
                    raise ValueError("Pareto shape and scale must be positive.")
        elif name == "t":
            if params.size > 0:
                if params[0] <= 0:
                    raise ValueError("Student-t degrees of freedom must be positive.")
                if params.size == 3 and params[2] <= 0:
                    raise ValueError("Student-t scale must be positive.")

        return name, params

    def _sample_y(self) -> np.ndarray:
        """Sample the univariate covariate vector Y of shape (n,)."""
        if self.y_distribution is None:
            values = np.arange(self.y_upper_bound + 1)
            return self.rng.choice(values, size=self.n, p=self.y_probabilities)

        name, params = self._parse_continuous_distribution(self.y_distribution)

        if name in {"gaussian", "normal"}:
            loc, scale = (0.0, 1.0) if params.size == 0 else params
            return self.rng.normal(loc=loc, scale=scale, size=self.n)
        if name == "cauchy":
            loc, scale = (0.0, 1.0) if params.size == 0 else params
            return loc + scale * self.rng.standard_cauchy(size=self.n)
        if name == "uniform":
            low, high = (0.0, 1.0) if params.size == 0 else params
            return self.rng.uniform(low=low, high=high, size=self.n)
        if name == "pareto":
            shape = 1.5 if params.size == 0 else params[0]
            scale = 1.0 if params.size <= 1 else params[1]
            # NumPy's pareto returns support [0, inf); rescale to [scale, inf).
            return scale * (self.rng.pareto(a=shape, size=self.n) + 1.0)
        if name == "t":
            df = 3 if params.size == 0 else params[0]
            loc, scale = (0.0, 1.0) if params.size == 1 else params[1:]
            return loc + scale * self.rng.standard_t(df=df, size=self.n)

        # This can only be reached if the distribution registry is changed.
        raise RuntimeError(f"Unhandled distribution: {name!r}.")

    def _sample_community_labels(self, y: np.ndarray) -> np.ndarray:
        """Sample SBM community labels Z from Y under the selected mechanism."""
        if self.sampling == "random":
            return self.rng.integers(0, self.k, size=self.n)

        if self.sampling == "identical":
            return y.astype(int, copy=True)

        if self.sampling == "correlated":
            z = y.astype(int, copy=True)
            # With a single community there is no alternative label to switch to.
            if self.k == 1:
                return z

            switch_mask = self.rng.random(self.n) > self.rho
            n_switch = int(switch_mask.sum())
            if n_switch:
                # Draw uniformly from labels excluding the original Y label.
                alternatives = self.rng.integers(0, self.k - 1, size=n_switch)
                original = z[switch_mask]
                z[switch_mask] = alternatives + (alternatives >= original)
            return z

        if self.sampling == "step":
            y_min = float(np.min(y))
            y_max = float(np.max(y))
            if np.isclose(y_min, y_max):
                return np.zeros(self.n, dtype=int)
            edges = np.linspace(y_min, y_max, self.k + 1)
            # Values at the upper endpoint belong to community k - 1.
            return np.clip(np.digitize(y, edges[1:-1], right=False), 0, self.k - 1)

        if self.sampling == "softmax":
            intercept = self._coerce_softmax_parameter(
                self.softmax_intercept, default=np.zeros(self.k), name="softmax_intercept"
            )
            slope = self._coerce_softmax_parameter(
                self.softmax_slope,
                default=np.linspace(-1.0, 1.0, self.k),
                name="softmax_slope",
            )
            logits = intercept[None, :] + y[:, None] * slope[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(probabilities, axis=1)
            uniforms = self.rng.random(self.n)[:, None]
            return (uniforms > cumulative).sum(axis=1)

        raise RuntimeError(f"Unhandled sampling mode: {self.sampling!r}.")

    def _coerce_softmax_parameter(
        self,
        parameter,
        *,
        default,
        name,
    ):
        if parameter is None:
            return default.astype(float, copy=True)

        values = np.asarray(parameter, dtype=float)
        if values.ndim == 0:
            return np.full(self.k, float(values))
        if values.ndim != 1 or values.size != self.k:
            raise ValueError(
                f"{name} must be a scalar or a length-{self.k} vector."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values.")
        return values

    def _generate_probability_matrix(self):
        """Sample a block-probability matrix honoring sparsity and assortativity.

        Assortativity must lie in [-1, 1]:
            -  1.0: strongly assortative
            -  0.0: no within/between preference
            - -1.0: strongly disassortative
        """
        if not -1.0 <= self.assortativity <= 1.0:
            raise ValueError(
                "assortativity must lie in [-1, 1], where negative values "
                "produce disassortative block structure."
            )

        density = 1.0 - self.sparsity_bias
        random_scale = self.rng.uniform(0.5, 1.5, size=(self.k, self.k))
        diagonal_mask = np.eye(self.k, dtype=bool)

        # Positive assortativity favors within-community edges.
        # Negative assortativity favors between-community edges.
        multipliers = np.where(
            diagonal_mask,
            1.0 + self.assortativity,
            1.0 - self.assortativity,
        )

        probs = np.clip(density * random_scale * multipliers, 0.0, 1.0)

        if not self.symmetric:
            probs = (probs + probs.T) / 2.0

        return probs

    def _sample_block_probs(self) -> np.ndarray:
        """Return either supplied or randomly generated SBM block probabilities."""
        if self.block_probs is not None:
            return self.block_probs.copy()
        return self._generate_probability_matrix()

    def _sample_latent_sbm_covariate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample and return Y, one-hot labels, and block probabilities.

        This mirrors the latent-sampling style of ``SBMGenerator`` while
        retaining the sampled values as instance attributes.
        """
        y = self._sample_y()
        z = self._sample_community_labels(y)
        community_assignment = np.eye(self.k, dtype=np.int8)[z]
        block_probs = self._sample_block_probs()

        self.Y = y
        self.z = z
        self.community_assignment = community_assignment
        self.sampled_block_probs = block_probs
        return y, community_assignment, block_probs

    def get_name(self) -> str:
        """Return a compact, descriptive sampler identifier."""
        y_spec = (
            self.y_distribution.replace(" ", "-")
            if self.y_distribution is not None
            else f"discrete0-{self.y_upper_bound}"
        )
        return (
            f"SBM_covariate_n{self.n}_k{self.k}_y{y_spec}_"
            f"sampling{self.sampling}_assort{self.assortativity}_"
            f"sparsity{self.sparsity_bias}"
        )
