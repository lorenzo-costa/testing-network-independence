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

    def __init__(self, k=None, rng=None, solver=None, **kwargs):
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
