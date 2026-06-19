import numpy as np
from scipy import stats
from scipy.special import expit, ndtr

from .helper_functions._latent_samplers import LatentSampler

class GaussianNetwork(LatentSampler):
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
    edge_var : float, optional
        Variance of the edges (default is 1).
    symmetric : bool, optional
        Whether the adjacency matrix should be symmetric (default is True).
    self_loops : bool, optional
        Whether to allow self-loops (default is False).
    sparsity_exponent: float
        Controls sparsity level by multiplying expected adj matrix by n^(-sparsity_exponent).
        Higher values = sparser.
    X : np.ndarray, optional
        Pre-specified latent positions for X. If provided, these will be used instead of sampling
    Z : np.ndarray, optional
        Pre-specified latent positions for Z. If provided, these will be used instead of sampling
    
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        n,
        k,
        edge_var=1,
        symmetric=True,
        self_loops=False,
        sparsity_exponent=0,
        rng=None,
        X=None,
        Z=None,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.default_rng()
        
        LatentSampler.__init__(self, n=n, k=k, rng=rng, **kwargs)
        
        self.edge_var = edge_var
        self.symmetric = symmetric
        self.self_loops = self_loops
        self.sparsity_exponent = sparsity_exponent
        
        self.X = X
        self.Z = Z
    
    def __repr__(self):
        return self.get_name() + f"(n={self.n}, k={self.k}, edge_var={self.edge_var}, "\
            f"symmetric={self.symmetric}, self_loops={self.self_loops}, sparsity_exponent={self.sparsity_exponent})"

    def get_name(self):
        return f"GaussianNetwork_" + self.sampler_name

    def generate(self):
        """Sample matrix and latent positions. Model definiton specifies options for:
        - latent postion type (SBM or general copula-based)
        - symmetric adjacency or not (default symmetric)
        - self loops or not (default no self loops)

        Returns
        -------
        out: dict
            dictionary with keys 'A', 'B' for adjacency matrices, 'Z', 'X' for the latent positions
        """
        if self.X is not None and self.Z is not None:
            Z, X = self.Z, self.X
        else:
            Z, X = self._sample_latent()

        expected_A = Z @ Z.T
        expected_B = X @ X.T
        
        if self.sparsity_exponent > 0:
            expected_A = expected_A * self.n**(-self.sparsity_exponent)
            expected_B = expected_B * self.n**(-self.sparsity_exponent)

        A = self.rng.normal(loc=expected_A, scale=self.edge_var)
        B = self.rng.normal(loc=expected_B, scale=self.edge_var)

        if self.self_loops is False:
            A[np.diag_indices_from(A)] = 0
            B[np.diag_indices_from(B)] = 0

        # Symmetrise
        if self.symmetric is True:
            A = (A + A.T) / 2
            B = (B + B.T) / 2

        out = {"A": A, "B": B, "Z": Z, "X": X}

        return out


class BernoulliNetwork(LatentSampler):
    """
    Network Data Generating Process using Bernoulli likelihood.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Dimensionality of the latent space.
    edge_var : float, optional
        Variance of the edges (default is 1).
    rng : np.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        n,
        k,        
        rng=None,
        symmetric=True,
        self_loops=False,
        rdpg=False,
        sparsity_exponent=0,
        X=None,
        Z=None,
        **kwargs,
    ):
        LatentSampler.__init__(self, n=n, k=k, rng=rng, **kwargs)

        self.symmetric = symmetric
        self.self_loops = self_loops
        self.sparsity_exponent = sparsity_exponent
        self.rdpg = rdpg
        self.X = X
        self.Z = Z

    def get_name(self):
        return f"BernoulliNetwork_" + self.sampler_name
    
    def __repr__(self):
        return self.get_name() + f"(n={self.n}, k={self.k}, rdpg={self.rdpg}, "\
            f"symmetric={self.symmetric}, self_loops={self.self_loops}, sparsity_exponent={self.sparsity_exponent})"

    def generate(self):
        """Sample matrix and latent positions. Model definiton specifies options for:
        - latent postion type (SBM, RDPG, or general copula-based)
        - symmetric adjacency or not (default symmetric)
        - self loops or not (default no self loops)

        Returns
        -------
        out: dict
            dictionary with keys 'A', 'B' for adjacency matrices, 'Z', 'X' for the latent positions
        """
        if self.X is not None and self.Z is not None:
            X, Z = self.X, self.Z
            expected_A = Z @ Z.T
            expected_B = X @ X.T
        else:
            Z, X = self._sample_latent()
            expected_A = Z @ Z.T
            expected_B = X @ X.T
            if self.rdpg:
                # sparsity applied directly to inner product, 
                if self.sparsity_exponent > 0:
                    expected_A = expected_A * np.log(self.n)**(-self.sparsity_exponent)
                    expected_B = expected_B * np.log(self.n)**(-self.sparsity_exponent)
            else:
                # apply logit link to get probabilities
                if self.sparsity_exponent > 0:
                    expected_A = expected_A * np.log(self.n)**(-self.sparsity_exponent)
                    expected_B = expected_B * np.log(self.n)**(-self.sparsity_exponent)
                    
                expected_A = expit(expected_A)
                expected_B = expit(expected_B)

        # to be safe clip in 0, 1
        expected_A = np.clip(expected_A, 0, 1)
        expected_B = np.clip(expected_B, 0, 1)

        try:
            if self.symmetric is True:
                # generate only lower half and then sum to ensure symmetry
                A = np.tril(self.rng.binomial(1, expected_A), k=-1)
                B = np.tril(self.rng.binomial(1, expected_B), k=-1)

                A = A + A.T
                B = B + B.T
            else:
                A = self.rng.binomial(1, expected_A)
                B = self.rng.binomial(1, expected_B)
        except ValueError as e:
            # safeguard for probs not on 0, 1
            print(f"Error generating samples: {e}")
            print(f"Expected probabilities (A): {expected_A}")
            print(f"Expected probabilities (B): {expected_B}")
            raise ValueError

        if self.self_loops is False:
            A[np.diag_indices_from(A)] = 0
            B[np.diag_indices_from(B)] = 0

        out = {"A": A, "B": B, "Z": Z, "X": X}

        return out
