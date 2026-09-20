from ._base_class import BaseEstimationMethod


class FitIndependent(BaseEstimationMethod):
    """Estimate latent positions for Y and every X network."""

    def __init__(
        self,
        d_y=None,
        d_x=None,
        rng=None,
        solver=None,
        use_true_latent=False,
        **kwargs,
    ):
        super().__init__(
            d_y=d_y,
            d_x=d_x,
            rng=rng,
            solver=solver,
            use_true_latent=use_true_latent,
        )

    def fit(self, data, **kwargs):
        """Estimate latent positions from ``A_Y`` and each network in ``A_X``.

        Parameters
        ----------
        data : dict or sequence of arrays
            Multiple-network data accepted by :class:`BaseEstimationMethod`.
        """

        self._process_input(data)

        # For consistency with test methods.
        self.pvalue = None
        self.reject_null = None
        self.test_stat_estimate = None

    def get_name(self):
        return "FitIndependent"
