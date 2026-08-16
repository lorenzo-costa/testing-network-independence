import numpy as np

from .linear_model_sampler import LinearModelGenerator
from .rdpg_sampler import RDPGGenerator


class LatentSampler:
    """Select and validate a latent-variable sampler.

    Parameters
    ----------
    n : int
        Number of samples (nodes).
    k : list of int
        Dimensions in the order ``[ky, kx[0], ..., kx[p - 1]]``.
    rdpg_distr : str, optional
        Select the RDPG generator when provided. Otherwise the linear-model
        generator is used.
    rng : np.random.Generator, optional
        Random number generator shared with the selected sampler.
    **kwargs
        Arguments forwarded to the selected sampler.
    """

    def __init__(
        self,
        n,
        k,
        rdpg_distr=None,
        rng=None,
        **kwargs,
    ):
        self._validate_dimensions(n, k)

        self.n = int(n)
        self.k = tuple(int(dimension) for dimension in k)
        self.ky = self.k[0]
        self.kx = self.k[1:]
        self.p = len(self.kx)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.rdpg_distr = rdpg_distr

        if rdpg_distr is None:
            self.sampler_type = "linear_model"
            self.latent_sampler = LinearModelGenerator(
                n=self.n,
                k=self.k,
                rng=self.rng,
                **kwargs,
            )
        else:
            self.sampler_type = "rdpg"
            self.latent_sampler = RDPGGenerator(
                n=self.n,
                k=self.k,
                rdpg_distr=rdpg_distr,
                rng=self.rng,
                **kwargs,
            )

        self.sampler_name = self.latent_sampler.get_name()
        self.X = None
        self.Y = None
        self.is_null = getattr(self.latent_sampler, "is_null", None)

    @staticmethod
    def _validate_dimensions(n, k):
        if n <= 0:
            raise ValueError("n must be positive")
        if not isinstance(k, (list, tuple)) or len(k) < 2:
            raise ValueError(
                "k must be a list or tuple containing ky followed by at least "
                "one X dimension"
            )
        for index, dimension in enumerate(k):
            if isinstance(dimension, bool) or not isinstance(
                dimension, (int, np.integer)
            ):
                raise TypeError(f"k[{index}] must be an integer")
            if dimension <= 0:
                raise ValueError(f"k[{index}] (= {dimension}) must be positive")

    def _sample_latent(self):
        """Sample and return ``(Y, X)`` with X represented as a list."""
        if self.sampler_type == "linear_model":
            Y, X = self.latent_sampler._sample_latent_linear()
        else:
            Y, X = self.latent_sampler._sample_latent_rdpg()

        Y = np.asarray(Y, dtype=float)
        if Y.shape != (self.n, self.ky):
            raise ValueError(
                f"Y must have shape ({self.n}, {self.ky}); got {Y.shape}"
            )

        if not isinstance(X, (list, tuple)):
            raise TypeError(
                "X must be a list or tuple containing one array per predictor block"
            )
        if len(X) != self.p:
            raise ValueError(
                f"X must contain {self.p} predictor blocks; got {len(X)}"
            )

        validated_X = []
        for index, (block, dimension) in enumerate(zip(X, self.kx)):
            block = np.asarray(block, dtype=float)
            expected_shape = (self.n, dimension)
            if block.shape != expected_shape:
                raise ValueError(
                    f"X[{index}] must have shape {expected_shape}; got {block.shape}"
                )
            validated_X.append(block)

        self.Y = Y
        self.X = validated_X
        self.is_null = getattr(self.latent_sampler, "is_null", None)
        return self.Y, self.X

    def sample_latent(self):
        """Public alias for :meth:`_sample_latent`."""
        return self._sample_latent()
