import numpy as np
from scipy import stats
from scipy.special import expit

# TODO:
# - extend gaussian network generation to more than two


class BaseDPG:
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def generate(self):
        raise NotImplementedError("Subclasses should implement this!")

    def name(self):
        raise NotImplementedError("Subclasses should implement this!")


class CopulaDGP:
    """Base class for Copula Data Generating Processes."""

    def __init__(self, rng):
        self.rng = rng

    def _generate_correlated_gaussians(self):
        """
        Sample the latent vector for Gaussian Copula model (i.e. q in the notes)
        using a Gaussian Copula with correlation matrix R.
        R has block structure [[I, sI], [sI, I]].

        We simulate this by generating z ~ N(0, I) and x = sz + sqrt(1-s^2)e.
        """

        # # shape: (n, k)
        # cov_matrix = np.block([
        #     [self.sigma * np.eye(self.k),  np.eye(self.k)],
        #     [np.eye(self.k), self.sigma * np.eye(self.k)]
        # ])
        # q = self.rng.multivariate_normal(np.zeros(2*self.k), cov_matrix, size=self.n)
        # qz, qx = q[:, :self.k], q[:, self.k:]

        # more efficient version uses the formula
        # z \sim N(0, I), x = s * z + sqrt(1-s^2) * e
        z = self.rng.normal(size=(self.n, self.k))
        e = self.rng.normal(size=(self.n, self.k))
        if self.sigma == 1:
            x = z
        elif self.sigma == -1:
            x = -z
        else:
            x = self.sigma * z + np.sqrt(1 - self.sigma**2) * e

        return z, x


class GaussianNetwork(BaseDPG, CopulaDGP):
    """
    Weighted network DGP with Gaussian weights on edges.

    Allows for arbitrary marginal distributions for the latent positions Z and X,
    while maintaining a gaussian copula correlation structure.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Dimensionality of the latent space.
    sigma : float
        Correlation parameter for the Gaussian copula (-1 to 1).
    marginals : dict, scipy.stats.rv_continuous
        Marginal distributions for the latent variables. If dict it should have 'z' and 'x' keys.
    edge_var : float, optional
        Variance of the edges (default is 1).
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(self, n, k, sigma, marginals=stats.norm, edge_var=1, rng=None, **args):
        # note here by multiple inheritance BaseDPG init will be called
        super().__init__(rng=rng)

        self.n = n
        self.k = k
        self.sigma = sigma
        self.edge_var = edge_var

        self._type_check(marginals)
        self.marginals = marginals

        if isinstance(marginals, dict):
            self.marginal_z = marginals.get("z", stats.norm)
            self.marginal_x = marginals.get("x", stats.norm)
        else:
            self.marginal_z = marginals
            self.marginal_x = marginals

    def name(self):
        return "GaussianNetwork"

    def generate(self):
        # sample latent gaussian vectors
        q_z, q_x = self._generate_correlated_gaussians()

        # convert to uniform via CDF of standard normal
        u_z = stats.norm.cdf(q_z)
        u_x = stats.norm.cdf(q_x)

        # apply inverse CDF of desired marginals
        Z = self.marginal_z.ppf(u_z)
        X = self.marginal_x.ppf(u_x)

        # generate adj matrices
        # note, for some marginals (eg heavy tails) the dot product can be very large,
        # maybe it is worth looking into ways of somehow normalising this
        expected_A = Z @ Z.T
        expected_B = X @ X.T

        A = self.rng.normal(loc=expected_A, scale=self.edge_var)
        B = self.rng.normal(loc=expected_B, scale=self.edge_var)

        # Symmetrise
        A = (A + A.T) / 2
        B = (B + B.T) / 2

        out = {"A": A, "B": B, "Z": Z, "X": X}

        return out

    def _type_check(self, marginals):
        # for now only continuous marginals allowed
        if isinstance(marginals, dict):
            # stats._distn_infrastructure.rv_continuous_frozen happens when feeding an instantiated obeject (e.g. stats.norm())
            if not isinstance(
                marginals.get("z", stats.norm),
                (stats.rv_continuous, stats._distn_infrastructure.rv_continuous_frozen),
            ):
                raise ValueError("Invalid marginal distributions")
            if not isinstance(
                marginals.get("x", stats.norm),
                (stats.rv_continuous, stats._distn_infrastructure.rv_continuous_frozen),
            ):
                raise ValueError("Invalid marginal distributions")
        else:
            if not isinstance(marginals, stats.rv_continuous):
                raise ValueError("Invalid marginal distribution")

    def __repr__(self):
        return (
            f"GaussianNetwork(n={self.n}, k={self.k}, sigma={self.sigma}, "
            f"edge_var={self.edge_var}, "
            f"marginal_z={self.marginal_z}, marginal_x={self.marginal_x})"
        )


class BernoulliNetwork(BaseDPG, CopulaDGP):
    """
    Network Data Generating Process using Bernoulli likelihood.

    Allows for arbitrary marginal distributions for the latent positions Z and X,
    while maintaining a gaussian copula correlation structure. Dot product of latent
    positions models the log-odds of entries of the adjacency matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Dimensionality of the latent space.
    sigma : float
        Correlation parameter for the Gaussian copula (-1 to 1).
    marginals : dict, scipy.stats.rv_continuous
        Marginal distributions for the latent variables. If dict it should have 'z' and 'x' keys.
    edge_var : float, optional
        Variance of the edges (default is 1).
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(self, n, k, sigma, marginals=stats.norm, edge_var=1, rng=None, **args):
        # note here by multiple inheritance BaseDPG init will be called
        super().__init__(rng=rng)

        self.n = n
        self.k = k
        self.sigma = sigma
        self.edge_var = edge_var

        self._type_check(marginals)
        self.marginals = marginals

        if isinstance(marginals, dict):
            self.marginal_z = marginals.get("z", stats.norm)
            self.marginal_x = marginals.get("x", stats.norm)
        else:
            self.marginal_z = marginals
            self.marginal_x = marginals

    def name(self):
        return "BernoulliNetwork"

    def generate(self):
        # sample latent gaussian vectors
        q_z, q_x = self._generate_correlated_gaussians()

        # convert to Uniform via CDF of standard normal (PIT)
        u_z = stats.norm.cdf(q_z)
        u_x = stats.norm.cdf(q_x)

        # apply Inverse CDF (PPF) to get desired marginals
        Z = self.marginal_z.ppf(u_z)
        X = self.marginal_x.ppf(u_x)

        # generate adj matrix using logistic function
        prob_A = expit(Z @ Z.T)
        prob_B = expit(X @ X.T)

        # generate lower half and the sum to ensure symmetry
        A = np.tril(self.rng.binomial(1, prob_A), k=-1)
        B = np.tril(self.rng.binomial(1, prob_B), k=-1)

        A = A + A.T
        B = B + B.T

        out = {"A": A, "B": B, "Z": Z, "X": X}
        return out

    def __repr__(self):
        return (
            f"BernoulliNetwork(n={self.n}, k={self.k}, sigma={self.sigma}, "
            f"edge_var={self.edge_var}, "
            f"marginal_z={self.marginal_z}, marginal_x={self.marginal_x})"
        )

    def _type_check(self, marginals):
        # for now only continuous marginals allowed
        if isinstance(marginals, dict):
            # stats._distn_infrastructure.rv_continuous_frozen happens when feeding an instantiated obeject (e.g. stats.norm())
            if not isinstance(
                marginals.get("z", stats.norm),
                (stats.rv_continuous, stats._distn_infrastructure.rv_continuous_frozen),
            ):
                raise ValueError("Invalid marginal distributions")
            if not isinstance(
                marginals.get("x", stats.norm),
                (stats.rv_continuous, stats._distn_infrastructure.rv_continuous_frozen),
            ):
                raise ValueError("Invalid marginal distributions")
        else:
            if not isinstance(marginals, stats.rv_continuous):
                raise ValueError("Invalid marginal distribution")
