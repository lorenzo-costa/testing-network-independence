
import numpy as np


from .helper_functions.metrics_functions import multivariate_ac_coefficient_permutation, rv_coefficient
from ._base_class import BaseMethod

class BaseEstimationMethod(BaseMethod):
    """Base class for estimation methods"""
    def __init__(self, rng=None, solver=None, k=None, 
                 use_true_latent=False, **kwargs):
        super().__init__()

        self.rng = rng if rng is not None else np.random.default_rng()
        if solver is None:
            raise ValueError("Solver must be provided")
        self.solver = solver
        self.k = k
        self.use_true_latent = use_true_latent
    
    def _process_input(self, data):
        """Utility function to extract and process input data, estimate latent
        positions if needed, and store results in the object."""

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid data format. Expected a dictionary with keys 'A', 'B'."
            )
        if self.use_true_latent:
            if "X" not in data.keys() or "Z" not in data.keys():
                raise ValueError(
                    "True latent positions must be provided when use_true_latent is True."
                )
            X = data["X"]
            Z = data["Z"]
            A = data.get("A", None)
            B = data.get("B", None)
            Xhat = X.copy()
            Zhat = Z.copy()
        else:
            if "estimated_X" not in data.keys():
                # need to estimate latent positions
                A = data.get("A", None)
                B = data.get("B", None)
                self.A = A
                self.B = B
                # true latent positions may not be provided
                X = data.get("X", None)
                Z = data.get("Z", None)

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

                Zhat = self.solver(A, k=self.k, rng=self.rng)[0]  # 0 is the xhat, 1 are the evalues
                Xhat = self.solver(B, k=self.k, rng=self.rng)[0]

            else:
                Zhat = data.get("estimated_Z")
                Xhat = data.get("estimated_X")
                Z = data.get("Z", None)
                X = data.get("X", None)
                A = data.get("A", None)
                B = data.get("B", None)

        self.A = A
        self.B = B
        self.X = X
        self.Z = Z
        self.Zhat = Zhat
        self.Xhat = Xhat

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
    def __init__(self, rng=None, solver=None, k=None, use_true_latent=False,
                 test_function=multivariate_ac_coefficient_permutation, **kwargs):
        super().__init__(k=k, rng=rng, solver=solver, use_true_latent=use_true_latent)
        
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
        self.test_stat_estimate = self.test_function(Y = self.X, Z = self.Zhat)
        self.pvalue = None
        self.reject_null = None
        
    def get_name(self):
        return "EstimateAC"