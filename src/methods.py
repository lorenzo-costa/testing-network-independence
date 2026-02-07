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
            'estimated_latent' : [self.Xhat, self.Zhat],
            'true_latent' : [self.X, self.Z]
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
        
        self.rv_distr = []
        
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

        rv_est = self.test_function(Zhat, Xhat)
        self.rv_est = rv_est
        # permute one of the observe networks, re-estimate latent positions and compute RV coefficient
        if self.permutation_type == 'observed':
            for _ in range(self.npermutations):
                perm = self.rng.permutation(A.shape[0])
                B_perm = B[perm][:, perm]
                Xhat_perm = self.solver(B_perm, k=self.k, rng=self.rng)[0]
                rv_perm = self.test_function(Zhat, Xhat_perm)
                self.rv_distr.append(rv_perm)
        # estimate latent positions once, permute them and compute rv_coefficient
        elif self.permutation_type == 'latent':
            for _ in range(self.npermutations):
                perm = self.rng.permutation(Zhat.shape[0])
                Xhat_perm = Xhat[perm, :]
                rv_perm = self.test_function(Zhat, Xhat_perm)
                self.rv_distr.append(rv_perm)
        
        # compute pvalue
        pvalue = np.mean([rv >= rv_est for rv in self.rv_distr])
        self.pvalue = pvalue
        self.reject_null = self.pvalue < self.alpha

        return

    def get_estimated(self):
        """Get fit results 
         
        Returns
        -------
        A dictionary with 'estimated_latent', 'true_latent', 'p-value', 'reject_null', and 'null' keys.
        """
        results = {
            'estimated_latent': [self.Xhat, self.Zhat],
            'true_latent': [self.X, self.Z],
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
            self.p_value = 1 - stats.chi2.cdf(chi, df=k**2)
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
            self.p_value = 1.0 - stats.f.cdf(F_stat, df1, df2)

        self.rejected = self.p_value < self.alpha
        return

    def get_estimated(self):
        """Get fit results 
         
        Returns
        -------
        A dictionary with 'estimated_latent', 'true_latent', 'p-value', 'reject_null', and 'null' keys.
        """
        results = {
            'estimated_latent': [self.Xhat, self.Zhat],
            'true_latent': [self.X, self.Z],
            'p-value' : self.pvalue,
            'reject_null' : self.reject_null,
            'null' : self.null
        }
        return results


class DiffusionCorrelation(BaseMethod):
    """
    Implementation of Diffusion Correlation algorithm for testing
    association between graph structure and nodal attributes.
    """

    def __init__(
        self,
        rng=None,
        k=None,
        test_method="mgc",
        sigma=None,
        n_permutations=1000,
        alpha=None,
        **kwargs,
    ):
        """
        Parameters:
        -----------
        max_t : int
            Maximum diffusion time steps to consider
        n_permutations : int
            Number of permutations for p-value computation
        """
        self.rng = np.random.default_rng() if rng is None else rng
        self.n_permutations = n_permutations
        self.k = k
        self.test_method = test_method
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
        """
        Step 4: Compute Euclidean distance matrix
        """
        return squareform(pdist(U, metric="euclidean"))

    def compute_dcorr(self, D1, D2):
        """
        Compute distance correlation between two distance matrices

        This is a simplified version - you may want to use the full
        distance correlation formula or mgc library
        """
        # Center the distance matrices
        n = D1.shape[0]

        # Double centering
        D1_centered = self._double_center(D1)
        D2_centered = self._double_center(D2)

        # Compute distance covariance
        dcov = np.sqrt(np.sum(D1_centered * D2_centered) / (n * n))

        # Compute distance variances
        dvar1 = np.sqrt(np.sum(D1_centered * D1_centered) / (n * n))
        dvar2 = np.sqrt(np.sum(D2_centered * D2_centered) / (n * n))

        # Distance correlation
        if dvar1 * dvar2 > 0:
            dcorr = dcov / np.sqrt(dvar1 * dvar2)
        else:
            dcorr = 0

        return dcorr

    def _double_center(self, D):
        """
        Double centering of distance matrix
        """
        n = D.shape[0]
        row_mean = np.mean(D, axis=1, keepdims=True)
        col_mean = np.mean(D, axis=0, keepdims=True)
        total_mean = np.mean(D)

        return D - row_mean - col_mean + total_mean

    def permutation_test(self, A, X, test_statistic):
        """
        Step 7: Compute p-value using permutation test

        Parameters:
        -----------
        A : array-like, shape (n, n)
            Adjacency matrix
        X : array-like, shape (n, p)
            Nodal attributes
        test_statistic : float
            Observed test statistic

        Returns:
        --------
        p_value : float
            Permutation p-value
        """
        n = A.shape[0]
        null_distribution = []

        for _ in range(self.n_permutations):
            # Permute nodal attributes
            perm_idx = np.random.permutation(n)
            X_perm = X[perm_idx, :]

            # Compute test statistic under null
            _, stats = self.test(A, X_perm, return_details=True)
            null_distribution.append(stats["mgc_star"])

        # Compute p-value
        null_distribution = np.array(null_distribution)
        p_value = np.mean(null_distribution >= test_statistic)

        return p_value, null_distribution

    def fit(self, data, **kwargs):
        """
        Main testing procedure

        Parameters:
        -----------
        data : array-like, shape (n, n)

        return_details : bool
            Whether to return detailed results

        Returns:
        --------
        p_value : float
            P-value for the test
        results : dict (if return_details=True)
            Dictionary containing detailed results
        """
        A, B, X, Z = data

        A_symm = (A + A.T) / 2
        B_symm = (B + B.T) / 2

        # Step 2: Compute normalized Laplacian
        A_laplacian = self.compute_normalized_laplacian(A_symm)
        B_laplacian = self.compute_normalized_laplacian(B_symm)

        # Step 3: Diffusion map for t=1 (i.e. ASE)
        Xhat, _ = ASE(A_laplacian, k=self.k)
        Zhat, _ = ASE(B_laplacian, k=self.k)
        Xhat = Xhat[0]
        Zhat = Zhat[0]

        distances_A = self.compute_distance_matrix(Xhat)
        distances_B = self.compute_distance_matrix(Zhat)

        if self.test_method == "mgc":
            random_state = self.rng.integers(100)

            p_value = stats.multiscale_graphcorr(
                distances_A,
                distances_B,
                compute_distance=None,
                random_state=random_state,
            )
        elif self.test_method == "dcorr":
            dcorr = self.compute_dcorr(distances_A, distances_B)
            p_value, null_dist = self.permutation_test(A, X, dcorr)
        else:
            print(self.test_method)
            raise ValueError("Unknown method for computing test statistic.")

        self.pvalue = p_value

        self.rejected = self.pvalue < self.alpha
        return self.Xhat, self.Zhat, X, Z

    def get_estimated(self):
        """Return true if the null hypothesis is rejected"""
        return self.rejected

    def get_truth(self):
        """Return True if the null hypothesis is False (i.e. should be rejected)"""
        return False if self.sigma == 0 else True


# class OMNITest(BaseMethod):
#     def __init__(self,
#                  sigma,
#                  alpha=0.05,
#                  rng=None,
#                  **args):
#         super().__init__()
#         self.sigma = sigma
#         self.alpha = alpha
#         self.rng = rng if rng is not None else np.random.default_rng()

#     def fit(self, A, B=None, k=2, *args, **kwargs):
#         """OMNI embedding with ase, approximate asymptotic distribution,
#         get p-value.
#         Assumes only two graphs

#         Need to figure our point 2
#         """
#         # build OMNI matrix
#         M = np.block([[A, (A+B)/2],
#                        [(A+B)/2, B]])

#         self.Mhat = ASE(M, k=k, rng=self.rng)[0][0]
#         self.Xhat = self.Mhat[:A.shape[0]]
#         self.Zhat = self.Mhat[A.shape[0]:]

#         return

#     def get_estimated(self):
#         """Return true if the null hypothesis is rejected"""
#         return self.rejected
#     def get_truth(self):
#         """Return True if the null hypothesis is False (i.e. should be rejected)"""
#         return False if self.sigma == 0 else True

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
