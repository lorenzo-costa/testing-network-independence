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
    eps_variance : float or array-like of shape (d_y, d_y), default=1
        Scalar variance or full covariance of zero-mean latent errors.
    b_mean, b_variance : float, default=0, 1
        Mean and nonnegative variance of independent coefficient entries.
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
        B=None,
        snr=None,
        x_mean=0,
        x_variance=1,
        eps_variance=1,
        b_mean=0,
        b_variance=1,
        x_distribution="multivariate_gaussian",
        eps_distribution="multivariate_gaussian",
        b_distribution="gaussian",
        edge_var=1,
        rng=None,
    ):
        self.edge_var = _finite_scalar(edge_var, "edge_var", nonnegative=True)
        self.latent_sampler = MultipleNetworksSampler(
            n=n,
            p=p,
            d_x=d_x,
            d_y=d_y,
            B=B,
            snr=snr,
            x_mean=x_mean,
            x_variance=x_variance,
            eps_variance=eps_variance,
            b_mean=b_mean,
            b_variance=b_variance,
            x_distribution=x_distribution,
            eps_distribution=eps_distribution,
            b_distribution=b_distribution,
            rng=rng,
        )
        self.rng = self.latent_sampler.rng
        self.n = self.latent_sampler.n
        self.p = self.latent_sampler.p
        self.d_x = self.latent_sampler.d_x
        self.d_y = self.latent_sampler.d_y
        self.snr = self.latent_sampler.snr

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
        upper = np.triu(sampled, k=1)
        return upper + upper.T

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
        return {
            "A_Y": self._sample_adjacency(latent["Y"]),
            "A_X": [self._sample_adjacency(x) for x in latent["X"]],
            "Y": latent["Y"],
            "X": latent["X"],
            "B": latent["B"],
        }


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
    eps_variance : float or array-like of shape (d_y, d_y), default=1
        Scalar variance or full covariance of zero-mean latent errors.
    b_mean, b_variance : float, default=0, 1
        Mean and nonnegative variance of independent coefficient entries.
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
        B=None,
        snr=None,
        x_mean=0,
        x_variance=1,
        eps_variance=1,
        b_mean=0,
        b_variance=1,
        x_distribution="multivariate_gaussian",
        eps_distribution="multivariate_gaussian",
        b_distribution="gaussian",
        rdpg=False,
        rng=None,
    ):
        if not isinstance(rdpg, (bool, np.bool_)):
            raise ValueError("rdpg must be a boolean.")
        self.rdpg = bool(rdpg)
        super().__init__(
            n=n,
            p=p,
            d_x=d_x,
            d_y=d_y,
            B=B,
            snr=snr,
            x_mean=x_mean,
            x_variance=x_variance,
            eps_variance=eps_variance,
            b_mean=b_mean,
            b_variance=b_variance,
            x_distribution=x_distribution,
            eps_distribution=eps_distribution,
            b_distribution=b_distribution,
            rng=rng,
        )

    def get_name(self):
        """Return the name of the multiple-network Bernoulli model."""
        return "BernoulliNetwork_multiple_networks"

    def __repr__(self):
        return (
            f"{self.get_name()}(n={self.n}, p={self.p}, d_x={self.d_x}, "
            f"d_y={self.d_y}, rdpg={self.rdpg})"
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
        upper = np.triu(sampled, k=1)
        return upper + upper.T
