import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
import pandas as pd
from .metrics import rv_coefficient
import sys
import os 
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) 

# TODO:
# - implement hoff llk ratio method 
# - implement OMNI method 

def solve_independent_old(A, k=2, rng=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()
    v0 = rng.normal(size=A.shape[0])
    evals, evectors = eigsh(A, k=k, which='LM', v0=v0)
    evals = np.clip(np.maximum(evals-0.5, 0), 0, 1e5) # clip for numerical stability
    xhat = evectors @ np.diag(np.sqrt(evals))
    return [xhat], [evals]

def ASE(A, k=2, rng=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()
    
    v0 = rng.standard_normal(size=A.shape[0])
    evals, evectors = eigsh(A, k=k, which='LM', v0=v0)
    evals = np.abs(evals-0.5)
    xhat = evectors * np.sqrt(evals)
    
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
        Xhat, evalsX = ASE(self.A, k=self.k, rng=self.rng)
        Zhat, evalsZ = ASE(self.B, k=self.k, rng=self.rng)
        
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
    def __init__(self, sigma, npermutations=100,
                 alpha=0.05, rng=None, **args):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.npermutations = npermutations
        self.rng = np.random.default_rng() if rng is None else rng

    def fit(self, A, B=None, k=2, *args, **kwargs):
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
        if B is None:
            if isinstance(A, (tuple, list)):
                A, B = A[0], A[1]
            else:
                raise ValueError("B must be provided if A is not a tuple of (A, B)")
        
        Xhat = ASE(A, k=k, rng=self.rng)[0][0]
        Zhat = ASE(B, k=k, rng=self.rng)[0][0]
        self.Xhat = Xhat
        self.Zhat = Zhat
        
        rv_est = rv_coefficient(Xhat, Zhat)
        self.rv_est = rv_est
        rv_distr = []
        for _ in range(self.npermutations):
            perm = self.rng.permutation(self.Xhat.shape[0])
            X_perm = self.Xhat[perm, :]
            rv_perm = rv_coefficient(X_perm, self.Zhat)
            rv_distr.append(rv_perm)
        self.rv_distr = rv_distr

        pvalue = np.mean([rv >= rv_est for rv in rv_distr])
        self.pvalue = pvalue
        self.rejected = self.pvalue < self.alpha

    def get_estimated(self):
        """Return true if the null hypothesis is rejected"""
        return self.rejected
    def get_truth(self):
        """
        FIXED: Return True if H1 is true (Signal exists), False if H0 is true.
        """
        # Old code: return True if self.sigma == 0 else False (BACKWARD)
        return False if self.sigma == 0 else True


class LLKRatioTest(BaseMethod):
    """Asymptotic Likelihood Ratio Test
    
    Adapted from Fosdick & Hoff (2015). Assume that the two networks have gaussian latent 
    positions, we want to test the null hypothesis that the cross-covariance matrix is zero 
    (i.e. independence for Gaussian).
    We estimate the latent positions using ASE (MLE in Gaussian case) and define the 
    LRT as the ratio of likelihood given the estimated latent positions:
    LRT = sup L_0(Sigma|X, Z)/sup L(Sigma|X, Z) = prod_{i=1}^k (1-r_i^2)^{-n/2}
    for r_i^2 the eigenvalues of (X^TX)^{-1/2}(X^TN)(N^TN)^{-1}(N^TX)(X^TX)^{-1/2}
 
    Parameters
    ----------
    sigma : float
        The true signal strength.
    alpha : float, optional
        The significance level (default 0.05)
    approximation : str, optional
        Approximation method to use for p-value computation. 
        Choices are:
        - beta: use the exact product of Beta distribution of the Wilks lambda
        Not implemented but requires some way of approximating quantiles 
        - chi-sq: use the chi-squared approximation
        - F-distr: use the F-distribution approximation
    """
    def __init__(self, 
                 sigma,
                 alpha=0.05, 
                 rng=None, 
                 approximation='beta', 
                 **args):
        
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.approximation = approximation
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def fit(self, 
            A, 
            B=None, 
            k=2, 
            **kwargs):
        """Estimates the latent positions and computes p-value

        Parameters
        ----------
        A : np.ndarray
            Adjacency matrix for first network
        B : np.ndarray
            Adjacency matrix for second network
        k : int, optional
            Number of dimensions for the latent space, by default 2
        """
        if B is None:
            if isinstance(A, (tuple, list)):
                A, B = A[0], A[1]
            else:
                raise ValueError("B must be provided if A is not a tuple of (A, B)")
        
        
        n = A.shape[0]
        
        # extract latent positions
        Xhat = ASE(A, k=k, rng=self.rng)[0][0]
        Zhat = ASE(B, k=k, rng=self.rng)[0][0]
        self.Xhat = Xhat
        self.Zhat = Zhat
        
        # # compute llk_score
        # cca_matrix = np.linalg.inv(Xhat.T @ Xhat) @ (Xhat.T @ Zhat) @ np.linalg.inv(Zhat.T @ Zhat) @ (Zhat.T @ Xhat)
        # cca_evals = np.linalg.eigvals(cca_matrix)
        # # llk_score = np.prod((1-cca_evals**2)**(-n/2))
        # # wilks score defined as llkratio**(2/n)
        # wilks_score = np.prod((1-cca_evals))
        
        # faster code
        Qx, _ = np.linalg.qr(Xhat)
        Qz, _ = np.linalg.qr(Zhat)
        S = np.linalg.svd(Qx.T @ Qz, compute_uv=False)
        cca_evals = S**2
        wilks_score = np.prod(1 - cca_evals)
        
        # approximate quantiles of the null 
        if self.approximation == 'beta':
            # use the exact product of beta distributions
            # TODO: find a way to approximate the quantiles
            raise NotImplementedError("Beta approximation not implemented yet")
        if self.approximation == 'chi-sq':
            # chi squared approximation -(n-1- (2k+1/2) log(llkratio^{2/n})\approx \chi^{2}_{k^{2}}
            chi = -(n-1-(2*k+1)/2) * np.log(wilks_score)
            self.p_value = 1-stats.chi2.cdf(chi, df=k**2)
        if self.approximation == 'F-distr':
            # define:
            # W : wilks score llk_ratio^{2/n}
            # a=\sqrt{(k^{4}-4)(2k^{2}-5) 
            # b=(n-2k-2)/2
            # df_{1}=u^{2}
            # df_{2}=u(n-2k+u-1)
            # then \frac{1-W^{1/u}}{W^{1/u}} (n-2k+u-1)(u)\approx F_{df_{1}, df_{2}}
            u = np.sqrt((k**4 - 4) / (2*k**2 - 5))
            df1 = k**2
            df2 = u * (n - 2*k + u - 1)
            W_u = wilks_score**(1.0 / u)
            F_stat = ((1.0 - W_u) / W_u) * ((n - 2*k + u - 1) / u)
            self.p_value = 1.0 - stats.f.cdf(F_stat, df1, df2)
        
        self.rejected = self.p_value < self.alpha
        return wilks_score

    def get_estimated(self):
        """Return true if the null hypothesis is rejected"""
        return self.rejected
    
    def get_truth(self):
        """Return True if the null hypothesis is False (i.e. should be rejected)"""
        return False if self.sigma == 0 else True
    
class OMNITest(BaseMethod):
    def __init__(self, 
                 sigma,
                 alpha=0.05, 
                 rng=None, 
                 **args):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, A, B=None, k=2, *args, **kwargs):
        """OMNI embedding with ase, approximate asymptotic distribution, 
        get p-value.
        Assumes only two graphs
        
        Need to figure our point 2
        """
        # build OMNI matrix
        M = np.block([[A, (A+B)/2],
                       [(A+B)/2, B]])
        
        self.Mhat = ASE(M, k=k, rng=self.rng)[0][0]
        self.Xhat = self.Mhat[:A.shape[0]]
        self.Zhat = self.Mhat[A.shape[0]:]
        
        return

    def get_estimated(self):
        """Return true if the null hypothesis is rejected"""
        return self.rejected
    def get_truth(self):
        """Return True if the null hypothesis is False (i.e. should be rejected)"""
        return False if self.sigma == 0 else True

# class FitDependent(BaseMethod):
#     # TODO: finish implementation of this
#     """Method to fit a shared embedding to both networks

#     Parameters
#     ----------
#     A: np.ndarray
#         Adjacency matrix for first network
#     B: np.ndarray
#         Adjacency matrix for second network
#     X: np.ndarray
#         Latent positions for first network
#     Z: np.ndarray
#         Latent positions for second network
#     """
#     def __init__(self, A, B, X=None, Z=None, rng=None):
#         super().__init__()
#         self.A = A
#         self.B = B
#         if A.shape != B.shape:
#             self.na = A.shape[0]
#             self.nb = B.shape[0]
#         else:
#             self.n = A.shape[0]
            
#         if X is not None:
#             self.X = X
#             self.k = X.shape[1]
#         if Z is not None:
#             self.Z = Z
#             self.k = Z.shape[1]
        
#         self.rng = rng if rng is not None else np.random.default_rng()

#     def fit(self, *args, **kwargs):
#         pass

#     def name(self):
#         return "DependentMethod"

#     def get_estimated(self):
#         pass

#     def get_truth(self):
#         return self.X, self.Z
