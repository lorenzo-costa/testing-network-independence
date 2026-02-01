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


class GaussianNetwork(BaseDPG):
    """
    Network Data Generating Process using Gaussian Copulas.
    
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
    marginal_z : scipy.stats.rv_continuous, optional
        Distribution for latent Z (default: standard normal).
    marginal_x : scipy.stats.rv_continuous, optional
        Distribution for latent X (default: standard normal).
    marginal_z_params : dict, optional
        Parameters for marginal_z (e.g., {'loc': 0, 'scale': 1}).
    marginal_x_params : dict, optional
        Parameters for marginal_x (e.g., {'df': 5} for t-distribution).
    edge_var : float, optional
        Variance of the edges (default is 1).
    rng : np.random.Generator, optional
        Random number generator.
    """
    def __init__(self, n, k, sigma, 
                 marginal_z=stats.norm, 
                 marginal_x=stats.norm,
                 marginal_z_params=None, 
                 marginal_x_params=None,
                 edge_var=1, rng=None,
                 **args):

        super().__init__(rng=rng)

        self.n = n
        self.k = k
        self.sigma = sigma
        self.edge_var = edge_var
        
        # Marginals
        self.marginal_z = marginal_z
        self.marginal_x = marginal_x
        self.marginal_z_params = marginal_z_params if marginal_z_params else {}
        self.marginal_x_params = marginal_x_params if marginal_x_params else {}
            
    def __repr__(self):
        return (f"GaussianNetwork(n={self.n}, k={self.k}, sigma={self.sigma}, "
                f"edge_var={self.edge_var}, "
                f"marginal_z={self.marginal_z}, marginal_x={self.marginal_x})")

    def name(self):
        return "GaussianNetwork"

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

    def generate(self):
        # sample latent gaussian vectors
        q_z, q_x = self._generate_correlated_gaussians()
        
        # convert to uniform via CDF of standard normal
        u_z = stats.norm.cdf(q_z)
        u_x = stats.norm.cdf(q_x)

        # apply inverse CDF of desired marginals
        Z = self.marginal_z.ppf(u_z, **self.marginal_z_params)
        X = self.marginal_x.ppf(u_x, **self.marginal_x_params)

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
        
        return A, B, Z, X

    # def generate_gaussian_data(self):
    #     I_k = np.eye(self.k)
    #     R = np.block([
    #             [I_k,          self.sigma * I_k],
    #             [self.sigma * I_k,  I_k        ]
    #         ])

    #     q = self.rng.multivariate_normal(np.zeros(2*self.k), R, size=self.n)
    #     Z = q[:, :self.k]
    #     X = q[:, self.k:]

    #     A = self.rng.normal(loc=Z @ Z.T, scale=self.edge_var)
    #     B = self.rng.normal(loc=X @ X.T, scale=self.edge_var)
    #     # symmetrise
    #     A = (A + A.T) / 2 
    #     B = (B + B.T) / 2
        
    #     return A, B, Z, X
    

class BernoulliNetwork(BaseDPG):
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
    marginal_z : scipy.stats.rv_continuous, optional
        Distribution for latent Z (default: standard normal).
    marginal_x : scipy.stats.rv_continuous, optional
        Distribution for latent X (default: standard normal).
    marginal_z_params : dict, optional
        Parameters for marginal_z (e.g., {'loc': 0, 'scale': 1}).
    marginal_x_params : dict, optional
        Parameters for marginal_x (e.g., {'df': 5} for t-distribution).
    edge_var : float, optional
        Variance of the edges (default is 1).
    rng : np.random.Generator, optional
        Random number generator.
    """
    def __init__(self, n, k, sigma, 
                 marginal_z=stats.norm, 
                 marginal_x=stats.norm,
                 marginal_z_params=None, 
                 marginal_x_params=None,
                 edge_var=1, rng=None,
                 **args):

        super().__init__(rng=rng)

        self.n = n
        self.k = k
        self.sigma = sigma
        self.edge_var = edge_var
        
        # Marginals
        self.marginal_z = marginal_z
        self.marginal_x = marginal_x
        self.marginal_z_params = marginal_z_params if marginal_z_params else {}
        self.marginal_x_params = marginal_x_params if marginal_x_params else {}
            
    def __repr__(self):
        return (f"GaussianNetwork(n={self.n}, k={self.k}, sigma={self.sigma}, "
                f"edge_var={self.edge_var}, "
                f"marginal_z={self.marginal_z}, marginal_x={self.marginal_x})")

    def name(self):
        return "GaussianNetwork"

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

    def generate(self):
        # sample latent gaussian vectors
        q_z, q_x = self._generate_correlated_gaussians()
        
        # convert to Uniform via CDF of standard normal (PIT)
        u_z = stats.norm.cdf(q_z)
        u_x = stats.norm.cdf(q_x)

        # apply Inverse CDF (PPF) to get desired marginals
        Z = self.marginal_z.ppf(u_z, **self.marginal_z_params)
        X = self.marginal_x.ppf(u_x, **self.marginal_x_params)
        
        # generate adj matrix using logistic function
        logit_A = Z @ Z.T
        logit_B = X @ X.T

        prob_A = expit(logit_A)
        prob_B = expit(logit_B)

        # generate lower half and the sum to ensure symmetry
        A = np.tril(self.rng.binomial(1, prob_A), k=-1)
        B = np.tril(self.rng.binomial(1, prob_B), k=-1)

        A = A + A.T
        B = B + B.T

        return A, B, Z, X