"""Covariate-aware stochastic block model sampler.

The public class name ``SBM_covarite_generator`` intentionally preserves the
spelling requested by the original API.  ``SBMCovariateGenerator`` is supplied
as a PEP-8-friendly alias.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np



class SBM_covarite_generator:
    """Sample an SBM together with a univariate node covariate ``X``.

    Parameters
    ----------
    n
        Number of nodes.
    k
        Number of SBM communities.
    x_distribution
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
    x_upper_bound
        Inclusive upper bound for a discrete covariate. A discrete covariate is
        sampled from ``{0, ..., x_upper_bound}``. Required when
        ``x_distribution`` is ``None``.
    x_probabilities
        Optional probability vector for values ``0, ..., x_upper_bound``.
        When omitted, the discrete covariate is sampled uniformly.
    sampling
        Relationship between X and SBM labels Z:

        - ``"random"``: Z is independent of X.
        - ``"identical"``: for discrete X, use ``Z = X``.
        - ``"correlated"``: for discrete X, retain ``Z = X`` with probability
          ``rho`` and otherwise switch to a uniformly selected different label.
        - ``"step"``: for continuous X, partition the observed range of X into
          k equal-width intervals and use the interval index as Z.
        - ``"softmax"``: for continuous X, sample Z using
          ``P(Z=j | X=x) ∝ exp(a_j + b_j x)``.
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
    directed
        Whether to sample a directed graph. Undirected graphs are symmetric and
        have no self-loops.
    self_loops
        Whether to permit self-loops. Ignored for undirected graphs unless set
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
        *,
        x_distribution = None,
        x_upper_bound = None,
        x_probabilities = None,
        sampling = "random",
        rho = 0.8,
        softmax_intercept = None,
        softmax_slope = None,
        block_probs = None,
        assortativity = 0.5,
        sparsity_bias = 0.6,
        directed = False,
        self_loops = False,
        rng = None,
    ) -> None:
        self.n = self._validate_positive_integer("n", n)
        self.k = self._validate_positive_integer("k", k)

        self.x_distribution = x_distribution
        self.x_upper_bound = x_upper_bound
        self.x_probabilities = x_probabilities
        self.sampling = str(sampling).lower()
        self.rho = float(rho)
        self.softmax_intercept = softmax_intercept
        self.softmax_slope = softmax_slope
        self.block_probs = block_probs
        self.assortativity = float(assortativity)
        self.sparsity_bias = float(sparsity_bias)
        self.directed = bool(directed)
        self.self_loops = bool(self_loops)
        self.rng = self._coerce_rng(rng)

        self._validate_parameters()
        self.covariate_type = (
            "continuous" if self.x_distribution is not None else "discrete"
        )

        # Convenience state populated by ``sample``.
        self.X  = None
        self.z: Optional[np.ndarray] = None
        self.community_assignment: Optional[np.ndarray] = None
        self.sampled_block_probs: Optional[np.ndarray] = None
        self.adjacency: Optional[np.ndarray] = None
        self.is_null = self.sampling == "random"

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> int:
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer; received {value!r}.")
        return int(value)

    @staticmethod
    def _coerce_rng(
        rng: Optional[Union[np.random.Generator, int]],
    ) -> np.random.Generator:
        if rng is None:
            return np.random.default_rng()
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng(rng)

    def _validate_parameters(self) -> None:
        if self.sampling not in self._SAMPLING_MODES:
            options = ", ".join(sorted(self._SAMPLING_MODES))
            raise ValueError(f"sampling must be one of {{{options}}}; got {self.sampling!r}.")

        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1].")
        if not 0.0 <= self.assortativity <= 1.0:
            raise ValueError("assortativity must lie in [0, 1].")
        if not 0.0 <= self.sparsity_bias <= 1.0:
            raise ValueError("sparsity_bias must lie in [0, 1].")

        if self.x_distribution is not None and self.x_upper_bound is not None:
            raise ValueError(
                "Specify either x_distribution (continuous X) or "
                "x_upper_bound (discrete X), not both."
            )
        if self.x_distribution is None and self.x_upper_bound is None:
            raise ValueError(
                "Specify x_distribution for continuous X or x_upper_bound "
                "for discrete X."
            )

        if self.x_distribution is not None:
            self._parse_continuous_distribution(self.x_distribution)
            if self.x_probabilities is not None:
                raise ValueError(
                    "x_probabilities is only valid for a discrete covariate."
                )
            if self.sampling in {"identical", "correlated"}:
                raise ValueError(
                    f'sampling="{self.sampling}" requires a discrete covariate '
                    "(use x_upper_bound)."
                )
        else:
            self.x_upper_bound = self._validate_nonnegative_integer(
                "x_upper_bound", self.x_upper_bound
            )
            if self.sampling in {"step", "softmax"}:
                raise ValueError(
                    f'sampling="{self.sampling}" requires a continuous covariate '
                    "(use x_distribution)."
                )
            if self.sampling in {"identical", "correlated"} and self.k != (
                self.x_upper_bound + 1
            ):
                raise ValueError(
                    'For sampling="identical" or "correlated", k must equal '
                    "x_upper_bound + 1 so every X value is a valid community "
                    "label."
                )
            if self.x_probabilities is not None:
                probs = np.asarray(self.x_probabilities, dtype=float)
                if probs.ndim != 1 or probs.size != self.x_upper_bound + 1:
                    raise ValueError(
                        "x_probabilities must be a 1D vector of length "
                        "x_upper_bound + 1."
                    )
                if np.any(probs < 0) or not np.isfinite(probs).all():
                    raise ValueError(
                        "x_probabilities must contain finite, nonnegative values."
                    )
                if not np.isclose(probs.sum(), 1.0):
                    raise ValueError("x_probabilities must sum to 1.")
                self.x_probabilities = probs

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
            if not self.directed and not np.allclose(block_probs, block_probs.T):
                raise ValueError(
                    "An undirected graph requires a symmetric block_probs matrix."
                )
            self.block_probs = block_probs

    @staticmethod
    def _validate_nonnegative_integer(name: str, value: int) -> int:
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 0:
            raise ValueError(
                f"{name} must be a nonnegative integer; received {value!r}."
            )
        return int(value)

    @classmethod
    def _parse_continuous_distribution(
        cls, specification: str
    ) -> tuple[str, np.ndarray]:
        if not isinstance(specification, str) or not specification.strip():
            raise ValueError("x_distribution must be a non-empty string.")

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
                "x_distribution string."
            ) from exc

        expected = {
            "gaussian": (0, 2),
            "normal": (0, 2),
            "cauchy": (0, 2),
            "uniform": (2,),
            "pareto": (1, 2),
            "t": (1, 3),
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
            if not params[0] < params[1]:
                raise ValueError(
                    "For uniform, the lower endpoint must be smaller than the "
                    "upper endpoint."
                )
        elif name == "pareto":
            if params[0] <= 0 or (params.size == 2 and params[1] <= 0):
                raise ValueError("Pareto shape and scale must be positive.")
        elif name == "t":
            if params[0] <= 0:
                raise ValueError("Student-t degrees of freedom must be positive.")
            if params.size == 3 and params[2] <= 0:
                raise ValueError("Student-t scale must be positive.")

        return name, params

    def _sample_x(self) -> np.ndarray:
        """Sample the univariate covariate vector X of shape (n,)."""
        if self.x_distribution is None:
            values = np.arange(self.x_upper_bound + 1)
            return self.rng.choice(values, size=self.n, p=self.x_probabilities)

        name, params = self._parse_continuous_distribution(self.x_distribution)

        if name in {"gaussian", "normal"}:
            loc, scale = (0.0, 1.0) if params.size == 0 else params
            return self.rng.normal(loc=loc, scale=scale, size=self.n)
        if name == "cauchy":
            loc, scale = (0.0, 1.0) if params.size == 0 else params
            return loc + scale * self.rng.standard_cauchy(size=self.n)
        if name == "uniform":
            low, high = params
            return self.rng.uniform(low=low, high=high, size=self.n)
        if name == "pareto":
            shape = params[0]
            scale = 1.0 if params.size == 1 else params[1]
            # NumPy's pareto returns support [0, inf); rescale to [scale, inf).
            return scale * (self.rng.pareto(a=shape, size=self.n) + 1.0)
        if name == "t":
            df = params[0]
            loc, scale = (0.0, 1.0) if params.size == 1 else params[1:]
            return loc + scale * self.rng.standard_t(df=df, size=self.n)

        # This can only be reached if the distribution registry is changed.
        raise RuntimeError(f"Unhandled distribution: {name!r}.")

    def _sample_community_labels(self, x: np.ndarray) -> np.ndarray:
        """Sample SBM community labels Z from X under the selected mechanism."""
        if self.sampling == "random":
            return self.rng.integers(0, self.k, size=self.n)

        if self.sampling == "identical":
            return x.astype(int, copy=True)

        if self.sampling == "correlated":
            z = x.astype(int, copy=True)
            # With a single community there is no alternative label to switch to.
            if self.k == 1:
                return z

            switch_mask = self.rng.random(self.n) > self.rho
            n_switch = int(switch_mask.sum())
            if n_switch:
                # Draw uniformly from labels excluding the original X label.
                alternatives = self.rng.integers(0, self.k - 1, size=n_switch)
                original = z[switch_mask]
                z[switch_mask] = alternatives + (alternatives >= original)
            return z

        if self.sampling == "step":
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            if np.isclose(x_min, x_max):
                return np.zeros(self.n, dtype=int)
            edges = np.linspace(x_min, x_max, self.k + 1)
            # Values at the upper endpoint belong to community k - 1.
            return np.clip(np.digitize(x, edges[1:-1], right=False), 0, self.k - 1)

        if self.sampling == "softmax":
            intercept = self._coerce_softmax_parameter(
                self.softmax_intercept, default=np.zeros(self.k), name="softmax_intercept"
            )
            slope = self._coerce_softmax_parameter(
                self.softmax_slope,
                default=np.linspace(-1.0, 1.0, self.k),
                name="softmax_slope",
            )
            logits = intercept[None, :] + x[:, None] * slope[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(probabilities, axis=1)
            uniforms = self.rng.random(self.n)[:, None]
            return (uniforms > cumulative).sum(axis=1)

        raise RuntimeError(f"Unhandled sampling mode: {self.sampling!r}.")

    def _coerce_softmax_parameter(
        self,
        parameter: Optional[ArrayLike],
        *,
        default: np.ndarray,
        name: str,
    ) -> np.ndarray:
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

    def _generate_probability_matrix(self) -> np.ndarray:
        """Sample a block matrix while honoring sparsity and assortativity."""
        density = 1.0 - self.sparsity_bias
        random_scale = self.rng.uniform(0.5, 1.5, size=(self.k, self.k))
        diagonal_mask = np.eye(self.k, dtype=bool)

        # At assortativity=0.5 both multipliers equal one.  Above it, diagonal
        # entries become larger relative to off-diagonal entries; below it, the
        # reverse holds.
        multipliers = np.where(
            diagonal_mask,
            2.0 * self.assortativity,
            2.0 * (1.0 - self.assortativity),
        )
        probs = np.clip(density * random_scale * multipliers, 0.0, 1.0)

        if not self.directed:
            probs = (probs + probs.T) / 2.0

        return probs

    def _sample_block_probs(self) -> np.ndarray:
        """Return either supplied or randomly generated SBM block probabilities."""
        if self.block_probs is not None:
            return self.block_probs.copy()
        return self._generate_probability_matrix()

    def _sample_adjacency(self, z: np.ndarray, block_probs: np.ndarray) -> np.ndarray:
        """Sample the adjacency matrix conditional on labels and block probabilities."""
        edge_probs = block_probs[z[:, None], z[None, :]]

        if self.directed:
            adjacency = (
                self.rng.random((self.n, self.n)) < edge_probs
            ).astype(np.int8)
            if not self.self_loops:
                np.fill_diagonal(adjacency, 0)
            return adjacency

        upper = self.rng.random((self.n, self.n)) < edge_probs
        adjacency = np.triu(upper, k=0 if self.self_loops else 1).astype(np.int8)
        adjacency = adjacency + np.triu(adjacency, k=1).T
        if not self.self_loops:
            np.fill_diagonal(adjacency, 0)
        return adjacency

    def sample(self) -> Dict[str, np.ndarray]:
        """Sample X, SBM labels, block probabilities, and the adjacency matrix.

        Returns
        -------
        dict
            ``{"X", "z", "community_assignment", "block_probs", "adjacency"}``.
            Here ``X`` and ``z`` have shape ``(n,)``,
            ``community_assignment`` has shape ``(n, k)``, ``block_probs`` has
            shape ``(k, k)``, and ``adjacency`` has shape ``(n, n)``.
        """
        x = self._sample_x()
        z = self._sample_community_labels(x)
        community_assignment = np.eye(self.k, dtype=np.int8)[z]
        block_probs = self._sample_block_probs()
        adjacency = self._sample_adjacency(z, block_probs)

        self.X = x
        self.z = z
        self.community_assignment = community_assignment
        self.sampled_block_probs = block_probs
        self.adjacency = adjacency

        return {
            "X": x,
            "z": z,
            "community_assignment": community_assignment,
            "block_probs": block_probs,
            "adjacency": adjacency,
        }

    def _sample_sbm_latent(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample and return X, one-hot labels, and block probabilities.

        This mirrors the latent-sampling style of ``SBMGenerator`` while
        retaining the sampled values as instance attributes.
        """
        x = self._sample_x()
        z = self._sample_community_labels(x)
        community_assignment = np.eye(self.k, dtype=np.int8)[z]
        block_probs = self._sample_block_probs()

        self.X = x
        self.z = z
        self.community_assignment = community_assignment
        self.sampled_block_probs = block_probs
        return x, community_assignment, block_probs

    def get_name(self) -> str:
        """Return a compact, descriptive sampler identifier."""
        x_spec = (
            self.x_distribution.replace(" ", "-")
            if self.x_distribution is not None
            else f"discrete0-{self.x_upper_bound}"
        )
        return (
            f"SBM_covariate_n{self.n}_k{self.k}_x{x_spec}_"
            f"sampling{self.sampling}_assort{self.assortativity}_"
            f"sparsity{self.sparsity_bias}"
        )


# PEP-8-friendly alias; the requested class name remains the primary API.
SBMCovariateGenerator = SBM_covarite_generator


__all__ = ["SBM_covarite_generator", "SBMCovariateGenerator"]