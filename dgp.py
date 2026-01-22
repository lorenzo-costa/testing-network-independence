import numpy as np
from scipy import stats

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


class GaussianNetwork:
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
        
        self.n = n
        self.k = k
        self.sigma = sigma
        self.edge_var = edge_var
        
        # Marginals
        self.marginal_z = marginal_z
        self.marginal_x = marginal_x
        self.marginal_z_params = marginal_z_params if marginal_z_params else {}
        self.marginal_x_params = marginal_x_params if marginal_x_params else {}
        
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def __repr__(self):
        return (f"GaussianNetwork(n={self.n}, k={self.k}, sigma={self.sigma}, "
                f"edge_var={self.edge_var}, "
                f"marginal_z={self.marginal_z}, marginal_x={self.marginal_x})")

    def name(self):
        return "GaussianNetwork"

    def _generate_correlated_gaussians(self):
        """
        Efficiently generates samples from N(0, R) without constructing 
        the full 2d x 2d matrix R.
        
        R has block structure [[I, sI], [sI, I]]. 
        We can simulate this by generating z ~ N(0, I) and x = sz + sqrt(1-s^2)e.
        """
        # 1. Generate independent standard normals
        # shape: (n, k)
        q_z = self.rng.standard_normal((self.n, self.k))
        noise = self.rng.standard_normal((self.n, self.k))
        
        # 2. Introduce correlation sigma for q_x
        if self.sigma == 1:
            q_x = q_z.copy()
        elif self.sigma == -1:
            q_x = -q_z.copy()
        else:
            scale = np.sqrt(1 - self.sigma**2)
            q_x = self.sigma * q_z + scale * noise
            
        return q_z, q_x

    def generate(self):
        # 1. Sample from the Gaussian Copula (Latent Gaussian Space)
        q_z, q_x = self._generate_correlated_gaussians()
        
        # 2. Convert to Uniform via CDF of standard normal (PIT)
        u_z = stats.norm.cdf(q_z)
        u_x = stats.norm.cdf(q_x)

        # 3. Apply Inverse CDF (PPF) of the desired marginals
        Z = self.marginal_z.ppf(u_z, **self.marginal_z_params)
        X = self.marginal_x.ppf(u_x, **self.marginal_x_params)
        
        # 4. Generate Adjacency Matrices based on dot products
        # note, for some marginals this can be very large, maybe it is worth looking
        # into ways of somehow standardising this
        expected_A = Z @ Z.T
        expected_B = X @ X.T
        
        A = self.rng.normal(loc=expected_A, scale=self.edge_var)
        B = self.rng.normal(loc=expected_B, scale=self.edge_var)
        
        # Symmetrise
        A = (A + A.T) / 2
        B = (B + B.T) / 2
        
        return A, B, Z, X