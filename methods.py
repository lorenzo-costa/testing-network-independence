import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
import pandas as pd
from metrics import rv_coefficient


def solve_independent(A, k=2, rng=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()
    v0 = rng.normal(size=A.shape[0])
    evals, evectors = eigsh(A, k=k, which='LM', v0=v0)
    evals = np.maximum(evals-0.5, 0)
    xhat = evectors @ np.diag(np.sqrt(evals))
    return [xhat], [evals]


class BaseMethod:
    def __init__(self):
        pass
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")
    def name(self):
        raise NotImplementedError("Subclasses should implement this!")
    def get_estimated(self):
        raise NotImplementedError("Subclasses should implement this!")
    def get_truth(self):
        raise NotImplementedError("Subclasses should implement this!")

class FitIndependent(BaseMethod):
    """Method to fit ase independently to each network

    Parameters
    ----------
    A: np.ndarray
        Adjacency matrix for first network
    B: np.ndarray
        Adjacency matrix for second network
    X: np.ndarray
        Latent positions for first network
    Z: np.ndarray
        Latent positions for second network
    """
    def __init__(self, A, B, X=None, Z=None, rng=None):
        super().__init__()
        self.A = A
        self.B = B
        if A.shape != B.shape:
            self.na = A.shape[0]
            self.nb = B.shape[0]
        else:
            self.n = A.shape[0]
            
        if X is not None:
            self.X = X
            self.k = X.shape[1]
        if Z is not None:
            self.Z = Z
            self.k = Z.shape[1]
        
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, *args, **kwargs):
        Xhat, evalsX = solve_independent(self.A, k=self.k, rng=self.rng)
        Zhat, evalsZ = solve_independent(self.B, k=self.k, rng=self.rng)
        
        self.Xhat = Xhat[0]
        self.Zhat = Zhat[0]

        return Xhat, Zhat

    def name(self):
        return "IndependentMethod"

    def get_estimated(self):
        return self.Xhat, self.Zhat

    def get_truth(self):
        return self.X, self.Z


class RVPermutationTest(BaseMethod):
    def __init__(self, A, B, sigma, rng=None):
        super().__init__()
        self.A = A
        self.B = B
        self.sigma = sigma
        self.rng = np.random.default_rng() if rng is None else rng

    def fit(self, k=2, nperm=100, *args, **kwargs):
        """Bootstrap the null distribution of the RV coefficient
        The function estimates the latent position of the networks independently
        using ASE, computes the RV coefficient and bootstraps the null distribution.

        Parameters
        ----------
        k : int, optional
            Number of dimensions for the latent space, by default 2
        nperm : int, optional
            Number of permutations for the bootstrap, by default 100
        """
        Xhat = solve_independent(self.A, k=k, rng=self.rng)[0][0]
        Zhat = solve_independent(self.B, k=k, rng=self.rng)[0][0]
        self.Xhat = Xhat
        self.Zhat = Zhat
        
        rv_est = rv_coefficient(Xhat, Zhat)
        self.rv_est = rv_est
        rv_distr = []
        for _ in range(nperm):
            perm = self.rng.permutation(self.Xhat.shape[0])
            X_perm = self.Xhat[perm, :]
            rv_perm = rv_coefficient(X_perm, self.Zhat)
            rv_distr.append(rv_perm)
        self.rv_distr = rv_distr

        pvalue = np.mean([rv > rv_est for rv in rv_distr])
        self.pvalue = pvalue
    
    def get_estimated(self):
        """Return estimated pvalue"""
        return self.pvalue
    def get_truth(self):
        """Return True if the null hypothesis is true, False otherwise"""
        return True if self.sigma == 0 else False


class FitDependent(BaseMethod):
    # TODO: finish implementation of this
    """Method to fit a shared embedding to both networks

    Parameters
    ----------
    A: np.ndarray
        Adjacency matrix for first network
    B: np.ndarray
        Adjacency matrix for second network
    X: np.ndarray
        Latent positions for first network
    Z: np.ndarray
        Latent positions for second network
    """
    def __init__(self, A, B, X=None, Z=None, rng=None):
        super().__init__()
        self.A = A
        self.B = B
        if A.shape != B.shape:
            self.na = A.shape[0]
            self.nb = B.shape[0]
        else:
            self.n = A.shape[0]
            
        if X is not None:
            self.X = X
            self.k = X.shape[1]
        if Z is not None:
            self.Z = Z
            self.k = Z.shape[1]
        
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, *args, **kwargs):
        pass

    def name(self):
        return "DependentMethod"

    def get_estimated(self):
        pass

    def get_truth(self):
        return self.X, self.Z
