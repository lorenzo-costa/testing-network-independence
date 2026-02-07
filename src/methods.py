import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
import pandas as pd
from .metrics import rv_coefficient, rv_coefficient_adjusted
import sys
import os
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
import numba as nb
from .solvers import ASE
from scipy.spatial.distance import pdist, squareform

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# TODO:
# - implement OMNI method

class BaseMethod:
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def name(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_estimated(self):
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

    def __init__(self, rng=None, solver=None, k=None, **kwargs):
        super().__init__()

        self.rng = rng if rng is not None else np.random.default_rng()
        if solver is None:
            raise ValueError("Solver must be provided")
        self.solver = solver
        self.k = k

    def fit(self, data, **kwargs):
        """Estimate latent positions independently

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        if not isinstance(data, dict):
            raise ValueError("Invalid data format. Expected a dictionary with keys 'A', 'B'.")

        A = data.get('A')
        B = data.get('B')
        # true latent positions may not be provided
        X = data.get('X', None)
        Z = data.get('Z', None)

        self.A = A
        self.B = B
        self.X = X
        self.Z = Z

        # get the number of dimensions (k). If X or Z is provided, use its 
        # shape (i.e. the "true" value of k)
        if X is not None or Z is not None:
            self.k = X.shape[1] if X is not None else Z.shape[1]
        else:
            if self.k is None:
                raise ValueError(
                    "Number of dimensions (k) must be specified if X and Z are not provided."
                )
            self.k = self.k

        Xhat, evalsX = self.solver(self.A, k=self.k, rng=self.rng)
        Zhat, evalsZ = self.solver(self.B, k=self.k, rng=self.rng)

        self.Xhat = Xhat
        self.Zhat = Zhat

        return

    def get_name(self):
        return "FitIndependent"

    def get_estimated(self):
        results = {
            'estimated_latent' : (self.Xhat, self.Zhat),
            'true_latent' : (self.X, self.Z)
        }
        return results


class RVPermutationTest(BaseMethod):
    """Perform RV permutation test for network independence.

    Parameters
    ----------
    sigma : float
        Correlation between latent positions (zero under independence i.e. H0 is true)
    npermutations : int
        Number of permutations to perform for the test.
    alpha : float
        Significance level for the test.
    solver : callable
        Function to estimate latent positions from the adjacency matrix.
    k : int
        Number of dimensions for the latent space.
    test_function : callable
        Function to compute the test statistic.
    permutation_type : str
        Type of permutation to use. Options:
        - latent: permute the estimated latent positions
        - observed: permute the observed data and re-estimate latent positions each time.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """
    def __init__(
        self,
        sigma,
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        k=None,
        test_function=rv_coefficient_adjusted,
        permutation_type='latent',
        **kwargs,
    ):
        super().__init__()
        
        self.sigma = sigma
        if sigma == 0:
            self.null = True
        else:
            self.null = False
        
        self.permutation_distribution = []
        
        self.alpha = alpha
        self.npermutations = npermutations
        self.rng = np.random.default_rng() if rng is None else rng
        
        if solver is None:
            raise ValueError("Solver must be provided")
        self.solver = solver
        
        self.k = k
        self.test_function = test_function
        self.permutation_type = permutation_type

    def fit(self, data, **kwargs):
        """Get null distribution of RV coefficient with permutations

        The function estimates the latent position of the networks independently,
        computes the RV coefficient and obtains the p-value by permutation.
        
        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """
        
        if not isinstance(data, dict):
            raise ValueError("Invalid data format. Expected a dictionary with keys 'A', 'B'.")

        A = data.get('A')
        B = data.get('B')
        self.A = A
        self.B = B
        # true latent positions may not be provided
        X = data.get('X', None)
        Z = data.get('Z', None)
        self.X = X
        self.Z = Z
        
        # get the number of dimensions (k). If X or Z is provided, use its 
        # shape (i.e. the "true" value of k)
        if X is not None or Z is not None:
            self.k = X.shape[1] if X is not None else Z.shape[1]
        else:
            if self.k is None:
                raise ValueError(
                    "Number of dimensions (k) must be specified if X and Z are not provided."
                )
            self.k = self.k
        
        Zhat = self.solver(A, k=self.k, rng=self.rng)[0] # 0 is the xhat, 1 are the evalues
        Xhat = self.solver(B, k=self.k, rng=self.rng)[0]
        
        self.Zhat = Zhat
        self.Xhat = Xhat

        test_stat_estimate = self.test_function(Zhat, Xhat)
        self.test_stat_estimate = test_stat_estimate
        # permute one of the observe networks, re-estimate latent positions and compute RV coefficient
        if self.permutation_type == 'observed':
            for _ in range(self.npermutations):
                perm = self.rng.permutation(A.shape[0])
                B_perm = B[perm][:, perm]
                Xhat_perm = self.solver(B_perm, k=self.k, rng=self.rng)[0]
                test_stat_perm = self.test_function(Zhat, Xhat_perm)
                self.permutation_distribution.append(test_stat_perm)
        # estimate latent positions once, permute them and compute rv_coefficient
        elif self.permutation_type == 'latent':
            for _ in range(self.npermutations):
                perm = self.rng.permutation(Zhat.shape[0])
                Xhat_perm = Xhat[perm, :]
                test_stat_perm = self.test_function(Zhat, Xhat_perm)
                self.permutation_distribution.append(test_stat_perm)

        # compute pvalue
        pvalue = np.mean([i >= self.test_stat_estimate for i in self.permutation_distribution])
        self.pvalue = bool(pvalue)
        self.reject_null = self.pvalue < self.alpha

        return

    def get_estimated(self):
        """Get fit results 
         
        Returns
        -------
        A dictionary with 'estimated_latent', 'true_latent', 'p-value', 'reject_null', and 'null' keys.
        """
        results = {
            'estimated_latent': (self.Xhat, self.Zhat),
            'true_latent': (self.X, self.Z),
            'p-value' : self.pvalue,
            'reject_null' : self.reject_null,
            'null' : self.null
        }
        return results

    def get_name(self):
        return "RVPermutationTest"


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
        Correlation between latent positions (zero under independence i.e. H0 is true)
    alpha : float, optional
        The significance level (default 0.05)
    approximation : str, optional
        Approximation method to use for p-value computation.
        Choices are:
        - beta: use the exact product of Beta distribution of the Wilks lambda
            Not implemented but requires some way of approximating quantiles
        - chi-sq: use the chi-squared approximation
        - F-distr: use the F-distribution approximation
    k : int, optional
        Number of dimensions for the latent space.
    solver : callable, optional
        Function to estimate the latent positions.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        sigma,
        alpha=0.05,
        approximation="beta",
        k=None,
        solver=None,
        rng=None,
        **args,
    ):
        super().__init__()
        
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        
        self.sigma = sigma
        if sigma == 0:
            self.null = True
        else:
            self.null = False
            
        self.alpha = alpha
        self.approximation = approximation
        self.k = k
        if solver is None:
            raise ValueError("Solver must be provided")
        self.solver = solver

    def fit(self, data, **kwargs):
        """Estimates the latent positions and computes p-value

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """
        
        if not isinstance(data, dict):
            raise ValueError("Invalid data format. Expected a dictionary with keys 'A', 'B'.")

        A = data.get('A')
        B = data.get('B')
        self.A = A
        self.B = B
        # true latent positions may not be provided
        X = data.get('X', None)
        Z = data.get('Z', None)
        self.X = X
        self.Z = Z

        # get the number of dimensions (k). If X or Z is provided, use its 
        # shape (i.e. the "true" value of k)
        if X is not None or Z is not None:
            self.k = X.shape[1] if X is not None else Z.shape[1]
        else:
            if self.k is None:
                raise ValueError(
                    "Number of dimensions (k) must be specified if X and Z are not provided."
                )
            self.k = self.k
        
        n = A.shape[0]
        k = self.k

        Zhat = self.solver(A, k=self.k, rng=self.rng)[0] # 0 is the xhat, 1 are the evalues
        Xhat = self.solver(B, k=self.k, rng=self.rng)[0]
        
        self.Zhat = Zhat
        self.Xhat = Xhat
        
        # # compute llk_score
        # cca_matrix = np.linalg.inv(Xhat.T @ Xhat) @ (Xhat.T @ Zhat) @ np.linalg.inv(Zhat.T @ Zhat) @ (Zhat.T @ Xhat)
        # cca_evals = np.linalg.eigvals(cca_matrix)
        # llk_score = np.prod((1-cca_evals**2)**(-n/2))
        # wilks score defined as llkratio**(2/n)
        # wilks_score = np.prod((1-cca_evals))

        # faster code
        Qx, _ = np.linalg.qr(Xhat)
        Qz, _ = np.linalg.qr(Zhat)
        S = np.linalg.svd(Qx.T @ Qz, compute_uv=False)
        cca_evals = S**2
        log_wilks = np.sum(np.log(1 - cca_evals + 1e-12))
        wilks_score = np.exp(log_wilks)

        # approximate quantiles of the null
        if self.approximation == "beta":
            # use the exact product of beta distributions
            # TODO: find a way to approximate the quantiles, Fosdick & Hoff use some monte carlo thing
            raise NotImplementedError("Beta approximation not implemented yet")
        if self.approximation == "chi-sq":
            # chi squared approximation -(n-1- (2k+1/2) log(llkratio^{2/n})\approx \chi^{2}_{k^{2}}
            chi = -(n - 1 - (2 * k + 1) / 2) * np.log(wilks_score)
            self.pvalue = 1 - stats.chi2.cdf(chi, df=k**2)
        if self.approximation == "F-distr":
            # define:
            # W : wilks score llk_ratio^{2/n}
            # a=\sqrt{(k^{4}-4)(2k^{2}-5)
            # b=(n-2k-2)/2
            # df_{1}=u^{2}
            # df_{2}=u(n-2k+u-1)
            # then \frac{1-W^{1/u}}{W^{1/u}} (n-2k+u-1)(u)\approx F_{df_{1}, df_{2}}
            a = np.sqrt((k**4 - 4) / (2 * k**2 - 5))
            b = n - 1 - (2 * k + 1) / 2
            df1 = k**2
            df2 = a * b - k**2 / 2 + 1
            W_a = wilks_score ** (1.0 / a)
            F_stat = ((1.0 - W_a) / W_a) * (a * b - k**2 / 2 + 1) / k**2
            self.pvalue = 1.0 - stats.f.cdf(F_stat, df1, df2)

        self.reject_null = self.pvalue < self.alpha
        return

    def get_estimated(self):
        """Get fit results 
         
        Returns
        -------
        A dictionary with 'estimated_latent', 'true_latent', 'p-value', 'reject_null', and 'null' keys.
        """
        results = {
            'estimated_latent': (self.Xhat, self.Zhat),
            'true_latent': (self.X, self.Z),
            'p-value' : self.pvalue,
            'reject_null' : self.reject_null,
            'null' : self.null
        }
        return results

class QAP(BaseMethod):
    """Quadratic Assignment Procedure

    Parameters
    ----------
    sigma : float
        Correlation between latent positions (zero under independence i.e. H0 is true)
    alpha : float, optional
        The significance level (default 0.05)
    npermutations : int, optional
        The number of permutations for the test (default 100)
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        sigma,
        alpha=0.05,
        npermutations=100,
        null_hypothesis='independence',
        rng=None,
        **args,
    ):
        super().__init__()
        
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        
        self.sigma = sigma
        if sigma == 0:
            self.null = True
        else:
            self.null = False
            
        self.alpha = alpha
        self.npermutations = npermutations
        self.null_hypothesis = null_hypothesis
        self.permutation_distribution = []

    def fit(self, data, **kwargs):
        """Estimates the latent positions and computes p-value

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B' i.e. adjacency matrices
        """
        
        if not isinstance(data, dict):
            raise ValueError("Invalid data format. Expected a dictionary with keys 'A', 'B'.")

        A = data.get('A')
        B = data.get('B')
        self.A = A
        self.B = B
        n = A.shape[0]

        self.test_stat_estimate = self._compute_test_stat(A, B)

        for i in range(self.npermutations):
            permutation = np.random.permutation(n)
            B_perm = B[permutation, :][:, permutation]
            test_stat_perm = self._compute_test_stat(A, B_perm)
            self.permutation_distribution.append(test_stat_perm)
        
        # compute pvalue
        pvalue = np.mean([i >= self.test_stat_estimate for i in self.permutation_distribution])
        self.pvalue = bool(pvalue)
        self.reject_null = self.pvalue < self.alpha

        return

    def get_estimated(self):
        """Get fit results 
         
        Returns
        -------
        A dictionary with 'estimated_latent', 'true_latent', 'p-value', 'reject_null', and 'null' keys.
        """
        results = {
            'p-value' : self.pvalue,
            'reject_null' : self.reject_null,
            'null' : self.null
        }
        return results    
    
    def _compute_test_stat(self, A, B):
        """Returns sqrt(n)rho if null hypothesis is independence (H0s) and sqrt(n)rho/v_w 
        if null hypothesis is un-correlated (H0w)"""
        
        n = A.shape[0]
        A_centered = A - A.mean(axis=0)
        B_centered = B - B.mean(axis=0)
        
        phi_0_hat = 1/(n*(n-1)-1) * np.sum(A_centered * B_centered)
        
        eta_hat_2_alpha = 1/(n*(n-1)-1) * np.sum(A_centered**2)
        eta_hat_2_beta = 1/(n*(n-1)-1) * np.sum(B_centered**2)
        
        # eta_hat_1_alpha = 1/n * np.sum((1/(n-1)*np.sum(A_centered * B_centered, axis = 1))**2)
        # eta_hat_1_beta = 1/n * np.sum((1/(n-1)*np.sum(B_centered * B_centered, axis = 1))**2)

        rho_hat = phi_0_hat / np.sqrt(eta_hat_2_alpha * eta_hat_2_beta)
        
        if self.null_hypothesis == 'independence':
            return np.sqrt(n) * rho_hat
        
        eta_hat_1_phi = 1/n * np.sum((1/(n-1)*np.sum(A_centered * B_centered, axis = 1))**2)
        
        v_w_hat = 4 * eta_hat_1_phi / (eta_hat_2_alpha * eta_hat_2_beta)

        return np.sqrt(n) * rho_hat / np.sqrt(v_w_hat)

class DiffusionCorrelation(BaseMethod):
    """Implementation of Diffusion Correlation algorithm.

    Parameters
    ----------
    sigma : float
        Correlation between latent positions (zero under independence i.e. H0 is true)
    k : int
        Dimensionality of the latent space.
    test_method : str
        Statistical test method to use. Options: "mgc", "dcorr".
    npermutations : int
        Number of permutations for significance testing.
    alpha : float
        Significance level for hypothesis testing.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """
    def __init__(
        self,
        sigma,
        k=None,
        test_method="mgc",
        npermutations=1000,
        alpha=0.05,
        rng=None,
        **kwargs,
    ):
        self.rng = np.random.default_rng() if rng is None else rng
        self.npermutations = npermutations
        self.k = k
        self.test_method = test_method
        
        if sigma == 0:
            self.null = True
        else:
            self.null = False
        self.sigma = sigma
        
        self.alpha = alpha

    def compute_normalized_laplacian(self, K):
        """
        Step 2: Compute normalized graph Laplacian
        L = B^(-1/2) * K * B^(-1/2)
        where B is the degree matrix
        """
        # Compute degree matrix
        degrees = np.sum(K, axis=1)
        # Avoid division by zero
        degrees[degrees == 0] = 1
        B_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

        # Normalized Laplacian
        L = B_inv_sqrt @ K @ B_inv_sqrt
        return L

    def compute_distance_matrix(self, U):
        return squareform(pdist(U, metric="euclidean"))

    def _double_center(self, D):
        """
        Double centering of distance matrix
        """
        n = D.shape[0]
        row_mean = np.mean(D, axis=1, keepdims=True)
        col_mean = np.mean(D, axis=0, keepdims=True)
        total_mean = np.mean(D)

        return D - row_mean - col_mean + total_mean

    def fit(self, data):
        """Compute p-value using permutation test

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """
        if not isinstance(data, dict):
            raise ValueError("Invalid data format. Expected a dictionary with keys 'A', 'B'.")

        A = data.get('A')
        B = data.get('B')
        self.A = A
        self.B = B
        # true latent positions may not be provided
        X = data.get('X', None)
        Z = data.get('Z', None)
        self.X = X
        self.Z = Z

        # get the number of dimensions (k). If X or Z is provided, use its 
        # shape (i.e. the "true" value of k)
        if X is not None or Z is not None:
            self.k = X.shape[1] if X is not None else Z.shape[1]
        else:
            if self.k is None:
                raise ValueError(
                    "Number of dimensions (k) must be specified if X and Z are not provided."
                )
            self.k = self.k

        A_symm = (A + A.T) / 2
        B_symm = (B + B.T) / 2

        # compute normalized Laplacian
        A_laplacian = self.compute_normalized_laplacian(A_symm)
        B_laplacian = self.compute_normalized_laplacian(B_symm)

        # diffusion map for t=1 (i.e. ASE)
        Xhat, _ = ASE(A_laplacian, k=self.k)
        Zhat, _ = ASE(B_laplacian, k=self.k)
        self.Xhat = Xhat
        self.Zhat = Zhat

        distances_A = self.compute_distance_matrix(Xhat)
        distances_B = self.compute_distance_matrix(Zhat)

        if self.test_method == "mgc":
            # random_state = self.rng.integers(100)

            p_value = stats.multiscale_graphcorr(
                distances_A,
                distances_B,
                compute_distance=None,
                random_state=self.rng,
            )
        # elif self.test_method == "dcorr":
        #     dcorr = self.compute_dcorr(distances_A, distances_B)
        #     p_value, null_dist = self.permutation_test(A, X, dcorr)
        else:
            print(self.test_method)
            raise ValueError("Unknown method for computing test statistic.")

        self.pvalue = p_value

        self.reject_null = self.pvalue < self.alpha
        return 

    def get_estimated(self):
        """Get fit results 
         
        Returns
        -------
        A dictionary with 'estimated_latent', 'true_latent', 'p-value', 'reject_null', and 'null' keys.
        """
        results = {
            'estimated_latent': (self.Xhat, self.Zhat),
            'true_latent': (self.X, self.Z),
            'p-value' : self.pvalue,
            'reject_null' : self.reject_null,
            'null' : self.null
        }
        return results



