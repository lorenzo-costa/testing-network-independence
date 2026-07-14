from ._base_class import BaseEstimationMethod


class FitIndependent(BaseEstimationMethod):
    """Estimate the latent positions of the single observed network."""

    def __init__(self, k=None, rng=None, solver=None, use_true_latent=False, **kwargs):
        super().__init__(
            k=k, rng=rng, solver=solver, use_true_latent=use_true_latent
        )

    def fit(self, data, **kwargs):
        """Estimate latent positions from ``A`` while retaining observed ``Y``.

        Parameters
        ----------
        data : dict
            A dictionary containing ``A``, observed ``Y``, and optionally true ``Z``.
        """

        self._process_input(data)

        # For consisetncy with test methods
        self.pvalue = None
        self.reject_null = None
        self.test_stat_estimate = None

    def get_name(self):
        return "FitIndependent"
