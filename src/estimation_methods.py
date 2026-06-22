
import numpy as np


from .helper_functions.metrics_functions import multivariate_ac_coefficient_permutation, rv_coefficient
from ._base_class import BaseEstimationMethod

class FitIndependent(BaseEstimationMethod):
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

    def __init__(self, 
                 k=None,
                 rng=None, 
                 solver=None, 
                 **kwargs):
        
        super().__init__(k=k, rng=rng, solver=solver)
    

    def fit(self, data, **kwargs):
        """Estimate latent positions independently

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        self._process_input(data)
        
        # For consisetncy with test methods
        self.pvalue = None
        self.reject_null = None
        self.test_stat_estimate = None

    def get_name(self):
        return "FitIndependent"


class EstimateRV(BaseEstimationMethod):
    """Method to return RV coefficient between the latent positions of two networks"""
    def __init__(self, rng=None, solver=None, k=None, use_true_latent=False,
                 test_function=rv_coefficient, **kwargs):
        super().__init__(k=k, rng=rng, solver=solver, use_true_latent=use_true_latent)

        self.test_function = test_function

    def fit(self, data, **kwargs):
        """Estimate RV coefficient between the latent positions of two networks

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        self._process_input(data)
        self.test_stat_estimate = self.test_function(self.Zhat, self.Xhat)
        self.pvalue = None
        self.reject_null = None

    def get_name(self):
        return "EstimateRV"

class EstimateAC(BaseEstimationMethod):
    """Method to return AC coefficient between the latent positions of two networks"""
    def __init__(self, rng=None, solver=None, k=None, use_true_latent=False, M=1,
                 test_function=multivariate_ac_coefficient_permutation, **kwargs):
        super().__init__(k=k, rng=rng, solver=solver, use_true_latent=use_true_latent)
        self.M = M
        self.test_function = test_function

    def fit(self, data, **kwargs):
        """Estimate AC coefficient between the latent positions of two networks

        Parameters
        ----------
        data : dict
            A dictionary containing keys 'A', 'B', 'X', 'Z' where 'A' and 'B' are adjacency matrices
            and 'X' and 'Z' are latent positions.
        """

        self._process_input(data)
        
        # X takes the role of response variable Y in the AC function, Z is the predictor.
        # for consistency keep the names as X and Z.
        # in _process_input when use_true_latent is True, Zhat copies true Z
        self.test_stat_estimate = self.test_function(Y = self.X, Z = self.Zhat, M = self.M, rng=self.rng)
        self.pvalue = None
        self.reject_null = None
        
    def get_name(self):
        return "EstimateAC"