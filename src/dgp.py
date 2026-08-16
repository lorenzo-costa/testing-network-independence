import numpy as np
from scipy.special import expit

from .latent_samplers import LatentSampler


class GaussianNetwork(LatentSampler):
    """Weighted network DGP"""

    def __init__(
        self,
        n,
        k,
        edge_var=1,
        symmetric=True,
        self_loops=False,
        sparsity_exponent=0,
        rng=None,
        **kwargs,
    ):
        """Weighted network DGP.

        Parameters
        ----------
        n : int
            Number of nodes in the network
        k : list of int
            Number of latent dimensions. First number is for Y, the rest are for X.
        edge_var : int, optional
            Variance of the edge weights, by default 1
        symmetric : bool, optional
            Whether the adjacency matrix should be symmetric, by default True
        self_loops : bool, optional
            Whether the adjacency matrix should have self-loops, by default False
        sparsity_exponent : int, optional
            Exponent for controlling sparsity, by default 0
        rng : _type_, optional
            Random number generator, by default None
        """

        rng = rng if rng is not None else np.random.default_rng()
        super().__init__(n=n, k=k, rng=rng, **kwargs)
        self.edge_var = edge_var
        self.symmetric = symmetric
        self.self_loops = self_loops
        self.sparsity_exponent = sparsity_exponent

    def __repr__(self):
        return (
            self.get_name()
            + f"(n={self.n}, k={self.k}, edge_var={self.edge_var}, "
            f"symmetric={self.symmetric}, self_loops={self.self_loops}, "
            f"sparsity_exponent={self.sparsity_exponent})"
        )

    def get_name(self):
        return "GaussianNetwork_" + self.sampler_name

    def _get_latent(self):
        Y, X = self._sample_latent()

        # Y needs to always be 2d array, X can contain multiple 2D arrays
        if Y.ndim != 2:
            raise ValueError(f"Y must be a 2D array, but got shape {Y.shape}")
        if not isinstance(X, list):
            raise ValueError(f"X must be a list of 2D arrays, but got type {type(X)}")
        for i, x in enumerate(X):
            if x.ndim != 2:
                raise ValueError(f"X[{i}] must be a 2D array, but got shape {x.shape}")

        Y = np.asarray(Y)
        self.Y = Y
        self.X = X

        return Y, X

    def generate(self):
        """Generate adjacency matrices.

        Returns
        -------
        dict
            Dictionary containing the generated adjacency matrices and latent variables.
            It has keys:
            - "Ay": Adjacency matrix for the Y latent variable.
            - "Ax": List of adjacency matrices for each X latent variable.
            - "Y": The latent variable Y.
            - "X": List of latent variables X.
        """

        Y, X = self._get_latent()

        expected_Ay = Y @ Y.T
        if self.sparsity_exponent > 0:
            expected_Ay *= self.n ** (-self.sparsity_exponent)

        Ay = self.rng.normal(loc=expected_Ay, scale=np.sqrt(self.edge_var))
        if not self.self_loops:
            Ay[np.diag_indices_from(Ay)] = 0
        if self.symmetric:
            Ay = (Ay + Ay.T) / 2

        out_X = []

        for i in range(len(X)):
            expected_Ax = X[i] @ X[i].T
            if self.sparsity_exponent > 0:
                expected_Ax *= self.n ** (-self.sparsity_exponent)

            Ax = self.rng.normal(loc=expected_Ax, scale=np.sqrt(self.edge_var))
            if not self.self_loops:
                Ax[np.diag_indices_from(Ax)] = 0
            if self.symmetric:
                Ax = (Ax + Ax.T) / 2
            out_X.append(Ax)

        return {"Ay": Ay, "Ax": out_X, "Y": Y, "X": X}


class BernoulliNetwork(GaussianNetwork):
    """Binary network DGP generated from latent-position inner products."""

    def __init__(
        self,
        n,
        k,
        rdpg=False,
        symmetric=True,
        self_loops=False,
        sparsity_exponent=0,
        rng=None,
        **kwargs,
    ):
        """Initialize a Bernoulli network generator.

        Parameters
        ----------
        n : int
            Number of nodes.
        k : list of int
            Latent dimensions in the order ``[ky, kx[0], ..., kx[p - 1]]``.
        rdpg : bool, default=False
            If True, inner products are treated directly as edge probabilities
            and clipped to ``[0, 1]``. Otherwise a logistic link is applied.
        symmetric : bool, default=True
            Whether to generate undirected adjacency matrices.
        self_loops : bool, default=False
            Whether diagonal edges are sampled.
        sparsity_exponent : float, default=0
            Exponent used in the Bernoulli sparsity scaling
            ``log(n) ** (-sparsity_exponent)``.
        rng : np.random.Generator, optional
            Random number generator shared with the latent sampler.
        **kwargs
            Arguments forwarded to :class:`GaussianNetwork` and the selected
            latent sampler.
        """
        super().__init__(
            n=n,
            k=k,
            rng=rng,
            symmetric=symmetric,
            self_loops=self_loops,
            sparsity_exponent=sparsity_exponent,
            **kwargs,
        )
        if not isinstance(rdpg, (bool, np.bool_)):
            raise TypeError("rdpg must be a boolean")
        self.rdpg = bool(rdpg)

    def get_name(self):
        return "BernoulliNetwork_" + self.sampler_name

    def __repr__(self):
        return (
            self.get_name()
            + f"(n={self.n}, k={self.k}, rdpg={self.rdpg}, "
            f"symmetric={self.symmetric}, self_loops={self.self_loops}, "
            f"sparsity_exponent={self.sparsity_exponent})"
        )

    def _edge_probabilities(self, latent):
        """Convert a latent-position matrix into an edge-probability matrix."""
        inner_products = latent @ latent.T

        if self.rdpg:
            if self.sparsity_exponent > 0:
                alpha = self.sparsity_exponent
                if not 0 <= alpha < 1:
                    raise ValueError(
                        "Use 0 <= sparsity_exponent < 1."
                    )
                rho_n = self.n ** (-alpha)
                probabilities = rho_n * inner_products
            else:
                probabilities = inner_products

            if probabilities.min() < 0 or probabilities.max() > 1:
                raise ValueError("Invalid RDPG probabilities.")

        else:
            # Logistic latent-position model - sparsity implemented as intercept on the logit scale
            alpha = self.sparsity_exponent
            if alpha > 0:
                rho_n = self.n ** (-alpha)
                intercept = np.log(rho_n)
            else:
                intercept = 0.0
            probabilities = expit(inner_products + intercept)

        return probabilities

    def _sample_adjacency(self, latent):
        """Sample one Bernoulli adjacency matrix from latent positions."""
        probabilities = self._edge_probabilities(latent)

        if self.symmetric:
            triangle = self.rng.binomial(1, probabilities)
            triangle = np.tril(triangle, k=0 if self.self_loops else -1)
            adjacency = triangle + triangle.T
            if self.self_loops:
                adjacency[np.diag_indices_from(adjacency)] //= 2
        else:
            adjacency = self.rng.binomial(1, probabilities)
            if not self.self_loops:
                adjacency[np.diag_indices_from(adjacency)] = 0

        return adjacency

    def generate(self):
        """Generate Bernoulli networks for Y and every X latent block.

        Returns
        -------
        dict
            ``Ay`` is the network generated from Y, ``Ax`` contains one network
            per X block, and ``Y`` and ``X`` contain the sampled latent positions.
        """
        Y, X = self._get_latent()

        Ay = self._sample_adjacency(Y)
        Ax = [self._sample_adjacency(block) for block in X]

        return {"Ay": Ay, "Ax": Ax, "Y": Y, "X": X}
