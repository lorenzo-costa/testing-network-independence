
import numpy as np


from .helper_functions.metrics_functions import multivariate_ac_coefficient_permutation, rv_coefficient
from ._base_class import BaseMethod


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

    def __init__(self, 
                 rng=None, 
                 solver=None, 
                 k=None,
                 **kwargs):
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

        self._process_input(data)

    def get_name(self):
        return "FitIndependent"


class EstimateRV(BaseMethod):
    """Method to return RV coefficient between the latent positions of two networks"""
    def __init__(self, rng=None, solver=None, k=None, 
                 test_function=rv_coefficient, **kwargs):
        super().__init__()

        self.rng = rng if rng is not None else np.random.default_rng()
        if solver is None:
            raise ValueError("Solver must be provided")
        self.solver = solver
        self.k = k
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
        self.test_stat = self.test_function(self.Zhat, self.Xhat)

    def get_name(self):
        return "EstimateRV"

class EstimateAC(BaseMethod):
    """Method to return AC coefficient between the latent positions of two networks"""
    def __init__(self, rng=None, solver=None, k=None, 
                 test_function=multivariate_ac_coefficient_permutation, **kwargs):
        super().__init__()

        self.rng = rng if rng is not None else np.random.default_rng()
        if solver is None:
            raise ValueError("Solver must be provided")
        self.solver = solver
        self.k = k
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
        self.test_stat = self.test_function(self.Zhat, self.Xhat)

    def get_name(self):
        return "EstimateAC"