import numpy as np
from scipy.special import expit

from .latent_samplers import MultipleNetworksSampler
from .latent_samplers.multiple_networks import _finite_scalar


class GaussianNetwork:
    """Generate p + 1 symmetric Gaussian networks with linearly related latents.

    Parameters
    ----------
    n : int
        Positive number of nodes shared by all networks.
    p : int
        Positive number of X networks, in addition to the Y network.
    d_x, d_y : int
        Positive latent dimensions of each X network and the Y network.
    B : {None, 0} or array-like of shape (d_y, p * d_x), optional
        Fixed coefficients in ``Y = concatenate(X, axis=1) @ B.T + epsilon``.
        Scalar 0 uses an all-zero matrix. If None or omitted, coefficients are
        sampled on every call to ``generate``.
    snr : float, optional
        Population latent signal/noise variance ratio, summed over Y dimensions.
        Rescale B to match snr, keeping eps_variance fixed. None disables scaling;
        zero gives zero coefficients. This controls latent noise, not edge_var.
    x_mean : float or array-like of shape (p * d_x,), default=0
        Mean of concatenated X positions, ordered by network.
    x_variance : float or array-like of shape (p * d_x, p * d_x), default=1
        Scalar variance times identity or full positive-semidefinite covariance.
    x_network_correlation : float, optional
        Correlation between matching dimensions in different X networks;
        different dimensions remain independent.
    eps_variance : float or array-like of shape (d_y, d_y), default=1
        Scalar variance or full covariance of zero-mean latent errors.
    b_mean, b_variance : float, default=0, 1
        Mean and nonnegative variance of independent coefficient entries.
    b_active_network_fraction : float, optional
        Fraction of sampled B network blocks to keep active. The rounded number
        of active networks is selected uniformly on every generation.
    x_distribution, eps_distribution : str, default="multivariate_gaussian"
        Registered distributions for X positions and latent errors.
    b_distribution : str, default="gaussian"
        Registered distribution for coefficient entries.
    edge_var : float, default=1
        Finite, nonnegative edge-error variance conditional on latent positions.
    rng : numpy.random.Generator, optional
        One random stream shared by latent sampling and all network edge draws.
    """

    def __init__(
        self,
        n,
        p,
        d_x,
        d_y,
        *,
        edge_var=1,
        rng=None,
        symmetric=True,
        allow_self_loops=False,
        latent_sampler=MultipleNetworksSampler,
        **kwargs):
        
        self.edge_var = _finite_scalar(edge_var, "edge_var", nonnegative=True)
        self.latent_sampler = latent_sampler(
            n=n,
            p=p,
            d_x=d_x,
            d_y=d_y,
            rng=rng,
            **kwargs
        )
        self.rng = self.latent_sampler.rng
        self.n = self.latent_sampler.n
        self.p = self.latent_sampler.p
        self.d_x = self.latent_sampler.d_x
        self.d_y = self.latent_sampler.d_y
        self.snr = getattr(self.latent_sampler, "snr", None)
        self.symmetric = symmetric
        self.allow_self_loops = allow_self_loops

    def __repr__(self):
        return (
            f"{self.get_name()}(n={self.n}, p={self.p}, d_x={self.d_x}, "
            f"d_y={self.d_y}, edge_var={self.edge_var})"
        )

    def get_name(self):
        """Return the name of the multiple-network Gaussian model."""
        return "GaussianNetwork_multiple_networks"

    def _sample_adjacency(self, latent):
        """Sample one Gaussian edge per upper-triangle entry and mirror it."""
        expected_A = latent @ latent.T
        sampled = self.rng.normal(loc=expected_A, scale=np.sqrt(self.edge_var))
        if self.symmetric:
            upper = np.triu(sampled, k=1)
            sampled = upper + upper.T

        # Remove self-loops if requested
        if not self.allow_self_loops:
            np.fill_diagonal(sampled, 0)

        return sampled
    
    def generate(self):
        """Sample latent positions and their symmetric, zero-diagonal networks.

        Returns
        -------
        dict
            Exactly ``A_Y`` of shape ``(n, n)``, ``A_X`` as a list of ``p``
            arrays of shape ``(n, n)``, ``Y`` of shape ``(n, d_y)``, ``X`` as
            a list of ``p`` arrays of shape ``(n, d_x)``, and ``B`` of shape
            ``(d_y, p * d_x)``. X networks preserve the latent block order.
        """
        latent = self.latent_sampler.sample_latent()
        if (
            hasattr(self.latent_sampler, "is_null")
            and getattr(self, "null_target", None) != "zero_covariance"
        ):
            self.is_null = self.latent_sampler.is_null
        result = {
            "A_Y": self._sample_adjacency(latent["Y"]),
            "A_X": [self._sample_adjacency(x) for x in latent["X"]],
            "Y": latent["Y"],
            "X": latent["X"],
        }
        if "B" in latent:
            result["B"] = latent["B"]
        return result


class BernoulliNetwork(GaussianNetwork):
    """Generate p + 1 symmetric Bernoulli networks with linearly related latents.

    Parameters
    ----------
    n : int
        Positive number of nodes shared by all networks.
    p : int
        Positive number of X networks, in addition to the Y network.
    d_x, d_y : int
        Positive latent dimensions of each X network and the Y network.
    B : {None, 0} or array-like of shape (d_y, p * d_x), optional
        Fixed coefficients in ``Y = concatenate(X, axis=1) @ B.T + epsilon``.
        Scalar 0 uses an all-zero matrix. If None or omitted, coefficients are
        sampled on each call to ``generate``.
    snr : float, optional
        Population latent signal/noise variance ratio, summed over Y dimensions.
        Rescale B to match snr, keeping eps_variance fixed. None disables scaling;
        zero gives zero coefficients. This precedes the logistic/RDPG edge link.
    x_mean : float or array-like of shape (p * d_x,), default=0
        Mean of concatenated X positions, ordered by network.
    x_variance : float or array-like of shape (p * d_x, p * d_x), default=1
        Scalar variance times identity or full positive-semidefinite covariance.
    x_network_correlation : float, optional
        Correlation between matching dimensions in different X networks;
        different dimensions remain independent.
    eps_variance : float or array-like of shape (d_y, d_y), default=1
        Scalar variance or full covariance of zero-mean latent errors.
    b_mean, b_variance : float, default=0, 1
        Mean and nonnegative variance of independent coefficient entries.
    b_active_network_fraction : float, optional
        Fraction of sampled B network blocks to keep active. The rounded number
        of active networks is selected uniformly on every generation.
    x_distribution, eps_distribution : str, default="multivariate_gaussian"
        Registered distributions for X positions and latent errors.
    b_distribution : str, default="gaussian"
        Registered distribution for coefficient entries.
    rdpg : bool, default=False
        If False, edge probabilities are ``expit(L @ L.T)`` for each latent
        matrix L. If True, probabilities are ``L @ L.T`` directly: every
        off-diagonal inner product must already lie in [0, 1]. Values are
        neither clipped nor rescaled. Latent positions themselves are unchanged.
    rng : numpy.random.Generator, optional
        One random stream shared by latent sampling and all network edge draws.

    Notes
    -----
    The inherited ``generate`` method returns exactly ``A_Y``, ``A_X``, ``Y``,
    ``X``, and ``B``, with the same dimensions as :class:`GaussianNetwork`.
    All adjacency matrices are symmetric and have zero diagonals.
    """

    def __init__(
            self,
            n,
            p,
            d_x,
            d_y,
            *,
            rdpg=False,
            rng=None,
            symmetric=True,
            latent_sampler=MultipleNetworksSampler,
            allow_self_loops=False,
            **kwargs):
        if not isinstance(rdpg, (bool, np.bool_)):
            raise ValueError("rdpg must be a boolean.")
        self.rdpg = bool(rdpg)
        super().__init__(
            n=n,
            p=p,
            d_x=d_x,
            d_y=d_y,
            rng=rng,
            symmetric=symmetric,
            latent_sampler=latent_sampler,
            allow_self_loops=allow_self_loops,
            **kwargs)

    def get_name(self):
        """Return the name of the multiple-network Bernoulli model."""
        return "BernoulliNetwork_multiple_networks"

    def __repr__(self):
        return (
            f"{self.get_name()}(n={self.n}, p={self.p}, d_x={self.d_x}, "
            f"d_y={self.d_y}, rdpg={self.rdpg}"
        )

    def _sample_adjacency(self, latent):
        """Sample Bernoulli edges from inner products and mirror the upper half."""
        probabilities = latent @ latent.T
        if not self.rdpg:
            probabilities = expit(probabilities)
            
        # Self-loops are absent, so squared latent norms need not be probabilities.
        np.fill_diagonal(probabilities, 0)
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            raise ValueError(
                "Edge probabilities must be finite and in [0, 1]. With rdpg=True, "
                "off-diagonal latent inner products must already lie in [0, 1]."
            )
        sampled = self.rng.binomial(1, probabilities)
        if self.symmetric:
            upper = np.triu(sampled, k=1)
            sampled = upper + upper.T

        # Remove self-loops if requested
        if not self.allow_self_loops:
            np.fill_diagonal(sampled, 0)

        return sampled
