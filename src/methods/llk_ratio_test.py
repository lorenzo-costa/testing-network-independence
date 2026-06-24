from scipy import stats
import numpy as np
from _base_class import BaseMethod


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


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
    rho: float
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
        k=None,
        alpha=0.05,
        approximation="beta",
        solver=None,
        rng=None,
        **args,
    ):
        super().__init__()

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        
        self.k = k

        self.alpha = alpha
        self.approximation = approximation
        
        if solver is None:
            raise ValueError("Solver must be provided")
        self.solver = solver

    def fit(self, data, **kwargs):
        """Estimates the latent positions and computes p-value

        ParametersFn
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary with keys 'A', 'B'."
            )

        if "estimated_Z" not in data.keys() or "estimated_X" not in data.keys():
            A = data.get("A")
            B = data.get("B")
            self.A = A
            self.B = B
            # true latent positions may not be provided
            X = data.get("X", None)
            Z = data.get("Z", None)
            self.X = X
            self.Z = Z

            # get the number of dimensions (k). If X or Z is provided, use its
            # shape (i.e. the "true" value of k)
            if X is not None or Z is not None:
                self.k = X.shape[1] if X is not None else Z.shape[1]

            Zhat = self.solver(A, k=self.k, rng=self.rng)[
                0
            ]  # 0 is the xhat, 1 are the evalues
            Xhat = self.solver(B, k=k, rng=self.rng)[0]
        else:
            Xhat = data.get("estimated_X")
            Zhat = data.get("estimated_Z")
            X = data.get("X", None)
            Z = data.get("Z", None)

        n, k = Zhat.shape
        self.Zhat = Zhat
        self.Xhat = Xhat
        self.X = X
        self.Z = Z

        # # compute llk_score
        # cca_matrix = np.linalg.inv(Xhat.T @ Xhat) @ (Xhat.T @ Zhat) @ np.linalg.inv(Zhat.T @ Zhat) @ (Zhat.T @ Xhat)
        # cca_evals = np.linalg.eigvals(cca_matrix)
        # llk_score = np.prod((1-cca_evals**2)**(-n/2))
        # wilks score defined as llkratio**(2/n)
        # wilks_score = np.prod((1-cca_evals))

        # faster code
        M_n = np.eye(n) - 1 / n * np.ones((n, n))
        Qx, _ = np.linalg.qr(M_n @ Xhat)
        Qz, _ = np.linalg.qr(M_n @ Zhat)
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

        self.reject_null = bool(self.pvalue < self.alpha)
        return

    def get_name(self):
        return "LLKRatioTest"

